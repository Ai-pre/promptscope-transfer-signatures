from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from src.activation.extractor import compute_delta_h, extract_hidden_states, get_positions, select_hidden_vector
from src.model.load_model import load_model
from src.prompt.prompt_builder import build_input
from src.utils.io import (
    ensure_base_prompt_record,
    ensure_dir,
    flatten_task_config,
    load_config,
    load_dataset,
    load_prompts,
    resolve_path,
    save_dataframe,
    set_seed,
    vector_rows_to_table,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract hidden-state activation deltas for prompts.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit-per-task", type=int, default=None)
    return parser.parse_args()


def aggregate_activation_rows(sample_rows):
    grouped_vectors = defaultdict(list)
    grouped_meta = {}

    group_columns = (
        "prompt_id",
        "group_id",
        "variant",
        "source",
        "prompt_text",
        "prompt_length_chars",
        "prompt_length_words",
        "task",
        "split",
        "layer",
        "position",
    )

    for row in sample_rows:
        key = tuple(row[column] for column in group_columns)
        grouped_vectors[key].append(np.asarray(row["delta_h"], dtype=np.float32))
        grouped_meta[key] = {column: row[column] for column in group_columns}

    summary_rows = []
    for key, vectors in grouped_vectors.items():
        record = dict(grouped_meta[key])
        record["num_samples"] = len(vectors)
        record["delta_h"] = np.mean(np.stack(vectors), axis=0)
        summary_rows.append(record)

    return summary_rows


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    set_seed(config.get("seed", 42))

    prompts_path = resolve_path(project_root, config["paths"]["prompts"])
    datasets_dir = resolve_path(project_root, config["paths"]["datasets_dir"])
    outputs_dir = ensure_dir(resolve_path(project_root, config["paths"]["outputs_dir"]) / "activations")

    prompts = ensure_base_prompt_record(
        load_prompts(prompts_path),
        base_prompt=config["base_prompt"],
    )
    task_names = flatten_task_config(config)
    datasets = {}
    for task_name in task_names:
        datasets[task_name] = load_dataset(
            datasets_dir / f"{task_name}.json",
            task_name=task_name,
            limit=args.limit_per_task or config.get("limit_per_task"),
        )

    model, tokenizer = load_model(
        model_name=config["model_name"],
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=config.get("device_map", "auto"),
    )

    sample_rows = []
    for prompt_record in prompts:
        for task_name, dataset in datasets.items():
            split = "seen" if task_name in config["tasks"]["seen"] else "unseen"
            for sample in dataset:
                prompt_text = build_input(tokenizer, prompt_record["text"], sample["input"])
                base_text = build_input(tokenizer, config["base_prompt"], sample["input"])

                hidden_prompt, _ = extract_hidden_states(model, tokenizer, prompt_text)
                hidden_base, _ = extract_hidden_states(model, tokenizer, base_text)

                prompt_positions = get_positions(
                    tokenizer=tokenizer,
                    rendered_prompt=prompt_text,
                    system_prompt=prompt_record["text"],
                    user_input=sample["input"],
                )
                base_positions = get_positions(
                    tokenizer=tokenizer,
                    rendered_prompt=base_text,
                    system_prompt=config["base_prompt"],
                    user_input=sample["input"],
                )

                for requested_layer in config["layers"]:
                    for position_name in config["positions"]:
                        prompt_layer, prompt_vector = select_hidden_vector(
                            hidden_states=hidden_prompt,
                            layer_spec=requested_layer,
                            token_index=prompt_positions[position_name],
                        )
                        _, base_vector = select_hidden_vector(
                            hidden_states=hidden_base,
                            layer_spec=requested_layer,
                            token_index=base_positions[position_name],
                        )
                        delta = compute_delta_h(prompt_vector, base_vector)

                        sample_rows.append(
                            {
                                "prompt_id": prompt_record["id"],
                                "group_id": prompt_record.get("group_id", prompt_record["id"]),
                                "variant": prompt_record.get("variant", "original"),
                                "source": prompt_record["source"],
                                "prompt_text": prompt_record["text"],
                                "prompt_length_chars": prompt_record.get("prompt_length_chars", len(prompt_record["text"])),
                                "prompt_length_words": prompt_record.get("prompt_length_words", len(prompt_record["text"].split())),
                                "task": task_name,
                                "split": split,
                                "sample_id": sample["id"],
                                "layer": prompt_layer,
                                "position": position_name,
                                "delta_h": delta.detach().float().cpu().numpy(),
                            }
                        )

    summary_rows = aggregate_activation_rows(sample_rows)

    sample_vectors, sample_meta = vector_rows_to_table(sample_rows, vector_key="delta_h")
    summary_vectors, summary_meta = vector_rows_to_table(summary_rows, vector_key="delta_h")

    np.savez_compressed(
        outputs_dir / "activation_vectors.npz",
        sample_vectors=sample_vectors,
        summary_vectors=summary_vectors,
    )
    save_dataframe(sample_meta, outputs_dir / "activation_metadata.parquet")
    save_dataframe(summary_meta, outputs_dir / "activation_summary.parquet")

    print(
        "Saved activation vectors to "
        f"{outputs_dir / 'activation_vectors.npz'} "
        f"with {len(sample_meta)} sample rows and {len(summary_meta)} summary rows."
    )


if __name__ == "__main__":
    main()

