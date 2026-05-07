from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import torch

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.activation.extractor import resolve_layer_index
from src.eval.evaluator import compute_accuracy, generate
from src.model.load_model import get_model_device, load_model
from src.prompt.prompt_builder import build_input, locate_token_positions
from src.utils.io import (
    ensure_base_prompt_record,
    ensure_dir,
    flatten_task_config,
    load_config,
    load_dataset,
    load_prompts,
    resolve_path,
    save_dataframe,
    save_json,
    save_markdown,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze early generation hidden-state trajectories for selected "
            "system prompts."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--prompt-ids",
        nargs="+",
        default=[
            "principle3_plain",
            "principle3_concise",
            "principle3_concise_format",
            "principle3_concise_careful_format",
        ],
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Defaults to config unseen tasks.")
    parser.add_argument("--limit-per-task", type=int, default=20)
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument("--generated-token-count", type=int, default=12)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-subdir", default="latent_trajectory_results")
    return parser.parse_args()


def find_prompt(prompts, prompt_id):
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            return prompt
    raise KeyError(f"Prompt id {prompt_id!r} not found.")


def hidden_forward(model, tokenizer, text):
    device = get_model_device(model)
    encoded = tokenizer(text, return_tensors="pt", add_special_tokens=False)
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.no_grad():
        outputs = model(**encoded, output_hidden_states=True)
    return outputs.hidden_states, encoded


def vector_norm(vector):
    return float(np.linalg.norm(np.asarray(vector, dtype=np.float64)))


def trajectory_metrics(vectors):
    vectors = np.asarray(vectors, dtype=np.float64)
    if len(vectors) == 0:
        return {
            "generated_tokens_observed": 0,
            "trajectory_path_length": float("nan"),
            "trajectory_endpoint_displacement": float("nan"),
            "trajectory_mean_step_norm": float("nan"),
            "trajectory_curvature_ratio": float("nan"),
            "first_generated_norm": float("nan"),
            "last_generated_norm": float("nan"),
        }
    if len(vectors) == 1:
        return {
            "generated_tokens_observed": 1,
            "trajectory_path_length": 0.0,
            "trajectory_endpoint_displacement": 0.0,
            "trajectory_mean_step_norm": 0.0,
            "trajectory_curvature_ratio": 0.0,
            "first_generated_norm": vector_norm(vectors[0]),
            "last_generated_norm": vector_norm(vectors[-1]),
        }
    steps = np.diff(vectors, axis=0)
    step_norms = np.linalg.norm(steps, axis=1)
    path_length = float(np.sum(step_norms))
    endpoint = float(np.linalg.norm(vectors[-1] - vectors[0]))
    return {
        "generated_tokens_observed": int(len(vectors)),
        "trajectory_path_length": path_length,
        "trajectory_endpoint_displacement": endpoint,
        "trajectory_mean_step_norm": float(np.mean(step_norms)),
        "trajectory_curvature_ratio": float(path_length / (endpoint + 1e-12)),
        "first_generated_norm": vector_norm(vectors[0]),
        "last_generated_norm": vector_norm(vectors[-1]),
    }


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return (
        frame.groupby(["prompt_id", "task", "split", "layer"], dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_prediction_words=("prediction_word_count", "mean"),
            trajectory_path_length=("trajectory_path_length", "mean"),
            trajectory_endpoint_displacement=("trajectory_endpoint_displacement", "mean"),
            trajectory_mean_step_norm=("trajectory_mean_step_norm", "mean"),
            trajectory_curvature_ratio=("trajectory_curvature_ratio", "mean"),
            first_user_token_norm=("first_user_token_norm", "mean"),
            system_last_token_norm=("system_last_token_norm", "mean"),
            num_samples=("correct", "count"),
        )
        .reset_index()
        .sort_values(["task", "layer", "prompt_id"])
    )


def format_report(summary_table):
    lines = ["# Latent Trajectory Analysis Report", "", "## Summary", ""]
    for row in summary_table.itertuples(index=False):
        lines.append(
            f"- {row.prompt_id} / {row.task} / layer={row.layer}: "
            f"acc={row.accuracy:.4f}, words={row.mean_prediction_words:.2f}, "
            f"path={row.trajectory_path_length:.4f}, endpoint={row.trajectory_endpoint_displacement:.4f}, "
            f"curvature={row.trajectory_curvature_ratio:.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    project_root = config["_project_root"]

    prompts_path = resolve_path(project_root, config["paths"]["prompts"])
    prompts = ensure_base_prompt_record(load_prompts(prompts_path), config["base_prompt"])
    selected_prompts = [find_prompt(prompts, prompt_id) for prompt_id in args.prompt_ids]

    datasets_dir = resolve_path(project_root, config["paths"]["datasets_dir"])
    task_names = args.tasks if args.tasks else list(config["tasks"]["unseen"])
    datasets = {
        task_name: load_dataset(
            datasets_dir / f"{task_name}.json",
            task_name=task_name,
            limit=args.limit_per_task,
        )
        for task_name in task_names
    }

    model, tokenizer = load_model(
        model_name=config["model_name"],
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=config.get("device_map", "auto"),
    )
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config["max_new_tokens"]

    layers = args.layers or config["layers"]
    sample_rows = []
    for prompt_record in selected_prompts:
        print(f"[latent_trajectory] prompt={prompt_record['id']}", flush=True)
        for task_name, dataset in datasets.items():
            split = "seen" if task_name in config["tasks"]["seen"] else "unseen"
            print(f"[latent_trajectory]   task={task_name} samples={len(dataset)}", flush=True)
            for sample in dataset:
                rendered_prompt = build_input(tokenizer, prompt_record["text"], sample["input"])
                prediction = generate(
                    model=model,
                    tokenizer=tokenizer,
                    prompt=rendered_prompt,
                    max_new_tokens=max_new_tokens,
                )
                correct, pred_norm, gold_norm = compute_accuracy(
                    prediction=prediction,
                    gold_label=sample["label"],
                    task_name=task_name,
                    sample=sample,
                )
                prompt_encoded = tokenizer(rendered_prompt, add_special_tokens=False)
                prompt_token_count = len(prompt_encoded["input_ids"])
                full_text = rendered_prompt + prediction
                hidden_states, full_encoded = hidden_forward(model, tokenizer, full_text)
                total_tokens = full_encoded["input_ids"].shape[1]
                generated_start = min(prompt_token_count, total_tokens)
                generated_end = min(generated_start + args.generated_token_count, total_tokens)

                positions = locate_token_positions(
                    tokenizer=tokenizer,
                    rendered_prompt=rendered_prompt,
                    system_prompt=prompt_record["text"],
                    user_input=sample["input"],
                )

                for requested_layer in layers:
                    layer_index = resolve_layer_index(hidden_states, requested_layer)
                    layer_states = hidden_states[layer_index][0].detach().float().cpu().numpy()
                    generated_vectors = layer_states[generated_start:generated_end]
                    metrics = trajectory_metrics(generated_vectors)
                    metrics.update(
                        {
                            "system_last_token_norm": vector_norm(layer_states[positions["system_last_token"]]),
                            "first_user_token_norm": vector_norm(layer_states[positions["first_user_token"]]),
                        }
                    )
                    sample_rows.append(
                        {
                            "prompt_id": prompt_record["id"],
                            "group_id": prompt_record.get("group_id", prompt_record["id"]),
                            "source": prompt_record["source"],
                            "prompt_text": prompt_record["text"],
                            "task": task_name,
                            "split": split,
                            "sample_id": sample["id"],
                            "input": sample["input"],
                            "label": str(sample["label"]),
                            "prediction": prediction,
                            "normalized_prediction": pred_norm,
                            "normalized_label": gold_norm,
                            "correct": correct,
                            "prediction_word_count": len(prediction.split()),
                            "layer": layer_index,
                            **metrics,
                        }
                    )

    sample_table = pd.DataFrame(sample_rows)
    summary_table = summarize(sample_rows)
    outputs_root = resolve_path(project_root, config["paths"]["outputs_dir"])
    results_dir = ensure_dir(outputs_root / args.output_subdir)
    save_dataframe(sample_table, results_dir / "latent_trajectory_samples.parquet")
    save_json(results_dir / "latent_trajectory_samples.json", sample_table.to_dict(orient="records"))
    save_dataframe(summary_table, results_dir / "latent_trajectory_summary.parquet")
    save_json(results_dir / "latent_trajectory_summary.json", summary_table.to_dict(orient="records"))
    save_markdown(results_dir / "latent_trajectory_report.md", format_report(summary_table))
    print(f"Saved latent trajectory outputs to {results_dir}")


if __name__ == "__main__":
    main()
