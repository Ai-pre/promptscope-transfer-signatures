from __future__ import annotations

import argparse
import copy
import json
from contextlib import contextmanager, nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.activation.extractor import resolve_layer_index
from src.eval.evaluator import compute_accuracy
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
            "Boundary-localized component-direction patching. Builds a direction "
            "such as concise - plain from activation summaries and patches it "
            "into selected token positions during generation."
        )
    )
    parser.add_argument("--config", required=True, help="Candidate/principle config.")
    parser.add_argument("--positive-prompt-id", default="principle3_concise")
    parser.add_argument("--negative-prompt-id", default="principle3_plain")
    parser.add_argument("--target-prompt-id", default="principle3_plain")
    parser.add_argument("--remove-from-prompt-id", default="principle3_concise")
    parser.add_argument("--direction-position", default="first_user_token")
    parser.add_argument("--layers", nargs="*", type=int, default=None)
    parser.add_argument(
        "--target-positions",
        nargs="+",
        default=[
            "system_last_token",
            "first_user_token",
            "user_middle_token",
            "user_last_token",
            "all_user_tokens",
            "first_generation_token",
        ],
    )
    parser.add_argument("--tasks", nargs="*", default=None, help="Defaults to config unseen tasks.")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--alphas", nargs="+", type=float, default=[1.0])
    parser.add_argument("--include-controls", dest="include_controls", action="store_true", default=True)
    parser.add_argument("--no-controls", dest="include_controls", action="store_false")
    parser.add_argument("--max-new-tokens", type=int, default=None)
    parser.add_argument("--output-subdir", default="component_patching_results")
    return parser.parse_args()


def get_nested_attr(obj, path: str):
    current = obj
    for part in path.split("."):
        if not hasattr(current, part):
            return None
        current = getattr(current, part)
    return current


def get_decoder_layers(model):
    candidates = [
        "model.layers",
        "model.model.layers",
        "language_model.model.layers",
        "language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
    ]
    for path in candidates:
        layers = get_nested_attr(model, path)
        if layers is not None:
            return layers
    raise ValueError("Could not locate decoder layers on this model.")


def resolve_model_layer(model, layer_spec: int):
    layers = get_decoder_layers(model)
    resolved = len(layers) + layer_spec if layer_spec < 0 else layer_spec
    if resolved < 0 or resolved >= len(layers):
        raise IndexError(f"Layer {layer_spec} resolved to {resolved}, but model has {len(layers)} layers.")
    return resolved, layers[resolved]


def load_activation_artifacts(config):
    outputs_root = resolve_path(config["_project_root"], config["paths"]["outputs_dir"])
    act_dir = outputs_root / "activations"
    results_dir = outputs_root / "results"
    return {
        "outputs_root": outputs_root,
        "activation_summary": pd.read_parquet(act_dir / "activation_summary.parquet"),
        "summary_vectors": np.load(act_dir / "activation_vectors.npz")["summary_vectors"],
        "slice_analysis": pd.read_parquet(results_dir / "slice_analysis.parquet"),
    }


def auto_layers_from_slice_analysis(artifacts):
    table = artifacts["slice_analysis"]
    candidates = table[table["slice_type"] == "layer_position"].copy()
    if candidates.empty:
        candidates = table.copy()
    ranked = candidates.sort_values(
        ["activation_ridge_r2", "activation_top_k_unseen_accuracy"],
        ascending=[False, False],
    )
    if ranked.empty or pd.isna(ranked.iloc[0].get("layer")):
        return []
    return [int(ranked.iloc[0]["layer"])]


def find_prompt(prompts, prompt_id):
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            return prompt
    raise KeyError(f"Prompt id {prompt_id!r} not found.")


def build_component_direction(
    *,
    activation_summary,
    summary_vectors,
    positive_prompt_id,
    negative_prompt_id,
    layer,
    position,
    tasks,
):
    table = activation_summary.copy()
    table["vector_row"] = table.index.to_numpy()
    table = table[
        (table["task"].isin(tasks))
        & (table["layer"] == layer)
        & (table["position"] == position)
        & (table["prompt_id"].isin([positive_prompt_id, negative_prompt_id]))
    ].copy()
    if table.empty:
        raise ValueError(
            f"No activation rows for prompts={positive_prompt_id},{negative_prompt_id}, "
            f"layer={layer}, position={position}."
        )

    diffs = []
    pair_tasks = []
    for task_name, group in table.groupby("task", dropna=False):
        pos = group[group["prompt_id"] == positive_prompt_id]
        neg = group[group["prompt_id"] == negative_prompt_id]
        if pos.empty or neg.empty:
            continue
        pos_vector = np.mean(summary_vectors[pos["vector_row"].to_numpy(dtype=int)], axis=0)
        neg_vector = np.mean(summary_vectors[neg["vector_row"].to_numpy(dtype=int)], axis=0)
        diffs.append(pos_vector - neg_vector)
        pair_tasks.append(task_name)
    if not diffs:
        raise ValueError("Could not build any positive-negative activation differences.")
    direction = np.mean(np.stack(diffs), axis=0).astype(np.float32)
    return direction, pair_tasks


def tokenizer_offsets(tokenizer, rendered_prompt):
    encoding = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return encoding["offset_mapping"]


def char_to_token(offsets, char_index):
    for token_index, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return token_index
    raise ValueError(f"Could not map character index {char_index} to token.")


def overlapping_tokens(offsets, start_char, end_char):
    tokens = []
    for token_index, (start, end) in enumerate(offsets):
        if end <= start_char or start >= end_char:
            continue
        tokens.append(token_index)
    return tokens


def resolve_patch_indices(tokenizer, rendered_prompt, system_prompt, user_input, position_name):
    encoded = tokenizer(rendered_prompt, add_special_tokens=False)
    seq_len = len(encoded["input_ids"])
    if position_name == "all_prompt_tokens":
        return list(range(seq_len))

    positions = locate_token_positions(
        tokenizer=tokenizer,
        rendered_prompt=rendered_prompt,
        system_prompt=system_prompt,
        user_input=user_input,
    )
    offsets = tokenizer_offsets(tokenizer, rendered_prompt)
    user_start, user_end = positions["user_char_span"]

    if position_name in {"system_last_token", "first_user_token"}:
        return [positions[position_name]]
    if position_name == "user_last_token":
        return [char_to_token(offsets, user_end - 1)]
    if position_name == "user_middle_token":
        user_tokens = overlapping_tokens(offsets, user_start, user_end)
        if not user_tokens:
            return [positions["first_user_token"]]
        return [user_tokens[len(user_tokens) // 2]]
    if position_name == "all_user_tokens":
        user_tokens = overlapping_tokens(offsets, user_start, user_end)
        return user_tokens or [positions["first_user_token"]]
    if position_name == "first_generation_token":
        return []
    raise ValueError(f"Unsupported target position: {position_name}")


def generation_kwargs_for_model(model, tokenizer, prompt_length, max_new_tokens):
    generation_config = getattr(model, "generation_config", None)
    if generation_config is not None:
        generation_config = copy.deepcopy(generation_config)
        generation_config.max_length = prompt_length + max_new_tokens
        generation_config.max_new_tokens = None
        generation_config.do_sample = False
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        for sampling_field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, sampling_field):
                setattr(generation_config, sampling_field, None)
        return {"generation_config": generation_config}
    return {
        "max_length": prompt_length + max_new_tokens,
        "do_sample": False,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }


@contextmanager
def patch_layer_context(model, *, layer, vector, alpha, target_position, token_indices):
    if vector is None or alpha == 0:
        yield
        return

    resolved_layer, module = resolve_model_layer(model, layer)
    state = {"call_index": 0}

    def hook(_module, _inputs, output):
        state["call_index"] += 1
        if isinstance(output, tuple):
            hidden = output[0]
            rest = output[1:]
        else:
            hidden = output
            rest = None
        if not torch.is_tensor(hidden) or hidden.dim() != 3:
            return output

        should_patch = False
        indices = token_indices
        if target_position == "first_generation_token":
            should_patch = state["call_index"] == 2
            indices = [hidden.shape[1] - 1]
        else:
            should_patch = state["call_index"] == 1 and bool(token_indices)

        if not should_patch:
            return output

        patch_vector = torch.as_tensor(vector, dtype=hidden.dtype, device=hidden.device)
        patched = hidden.clone()
        patched[:, indices, :] = patched[:, indices, :] + float(alpha) * patch_vector
        if rest is None:
            return patched
        return (patched, *rest)

    handle = module.register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def generate_one(
    *,
    model,
    tokenizer,
    rendered_prompt,
    max_new_tokens,
    layer=None,
    vector=None,
    alpha=0.0,
    target_position=None,
    token_indices=None,
):
    device = get_model_device(model)
    encoded = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]
    kwargs = generation_kwargs_for_model(model, tokenizer, prompt_length, max_new_tokens)

    context = patch_layer_context(
        model,
        layer=layer,
        vector=vector,
        alpha=alpha,
        target_position=target_position,
        token_indices=token_indices or [],
    ) if layer is not None and vector is not None else nullcontext()

    with torch.no_grad():
        with context:
            output_ids = model.generate(**encoded, **kwargs)
    generated = output_ids[0, prompt_length:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def matched_random_direction(direction, rng):
    random_vector = rng.normal(size=direction.shape).astype(np.float32)
    norm = np.linalg.norm(random_vector)
    if norm == 0:
        return random_vector
    return random_vector / norm * np.linalg.norm(direction)


def orthogonal_random_direction(direction, rng):
    random_vector = matched_random_direction(direction, rng)
    denom = float(np.dot(direction, direction))
    if denom == 0:
        return random_vector
    random_vector = random_vector - (float(np.dot(random_vector, direction)) / denom) * direction
    norm = np.linalg.norm(random_vector)
    if norm == 0:
        return matched_random_direction(direction, rng)
    return random_vector / norm * np.linalg.norm(direction)


def evaluate_condition(
    *,
    model,
    tokenizer,
    prompt_record,
    dataset,
    task_name,
    split,
    layer,
    vector,
    alpha,
    target_position,
    max_new_tokens,
):
    rows = []
    for sample in dataset:
        rendered_prompt = build_input(tokenizer, prompt_record["text"], sample["input"])
        token_indices = (
            resolve_patch_indices(
                tokenizer,
                rendered_prompt,
                prompt_record["text"],
                sample["input"],
                target_position,
            )
            if target_position is not None
            else []
        )
        prediction = generate_one(
            model=model,
            tokenizer=tokenizer,
            rendered_prompt=rendered_prompt,
            max_new_tokens=max_new_tokens,
            layer=layer,
            vector=vector,
            alpha=alpha,
            target_position=target_position,
            token_indices=token_indices,
        )
        correct, pred_norm, gold_norm = compute_accuracy(
            prediction=prediction,
            gold_label=sample["label"],
            task_name=task_name,
            sample=sample,
        )
        rows.append(
            {
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
            }
        )
    return rows


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    group_columns = [
        "condition",
        "task",
        "split",
        "layer",
        "direction_position",
        "target_position",
        "alpha",
        "patch_prompt_id",
        "direction_prompt_id",
        "control_type",
    ]
    return (
        frame.groupby(group_columns, dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_prediction_words=("prediction_word_count", "mean"),
            num_samples=("correct", "count"),
        )
        .reset_index()
        .sort_values(["task", "condition", "layer", "target_position", "alpha"])
    )


def format_report(summary_table, metadata):
    lines = [
        "# Component Patching Report",
        "",
        "## Direction",
        "",
        f"- Positive prompt: {metadata['positive_prompt_id']}",
        f"- Negative prompt: {metadata['negative_prompt_id']}",
        f"- Direction position: {metadata['direction_position']}",
        f"- Direction tasks: {', '.join(metadata['direction_tasks'])}",
        "",
        "## Results",
        "",
    ]
    for row in summary_table.itertuples(index=False):
        lines.append(
            f"- {row.task} / {row.condition} / layer={row.layer} / pos={row.target_position} / "
            f"alpha={row.alpha}: acc={row.accuracy:.4f}, words={row.mean_prediction_words:.2f}, n={row.num_samples}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_seed(config.get("seed", 42))
    rng = np.random.default_rng(config.get("seed", 42))
    project_root = config["_project_root"]
    artifacts = load_activation_artifacts(config)

    prompts_path = resolve_path(project_root, config["paths"]["prompts"])
    prompts = ensure_base_prompt_record(load_prompts(prompts_path), config["base_prompt"])
    positive_prompt = find_prompt(prompts, args.positive_prompt_id)
    negative_prompt = find_prompt(prompts, args.negative_prompt_id)
    target_prompt = find_prompt(prompts, args.target_prompt_id)
    removal_prompt = find_prompt(prompts, args.remove_from_prompt_id)

    datasets_dir = resolve_path(project_root, config["paths"]["datasets_dir"])
    task_names = args.tasks if args.tasks else list(config["tasks"]["unseen"])
    datasets = {
        task_name: load_dataset(
            datasets_dir / f"{task_name}.json",
            task_name=task_name,
            limit=args.limit_per_task if args.limit_per_task is not None else config.get("limit_per_task"),
        )
        for task_name in task_names
    }

    layers = args.layers or auto_layers_from_slice_analysis(artifacts)
    if not layers:
        layers = [int(layer) for layer in config["layers"]]
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else config["max_new_tokens"]

    model, tokenizer = load_model(
        model_name=config["model_name"],
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=config.get("device_map", "auto"),
    )
    # Resolve negative layer specs against actual model layers. Activation
    # summaries store resolved hidden-state layers, which include embedding at 0.
    num_hidden_states = len(get_decoder_layers(model)) + 1

    sample_rows = []
    metadata_rows = []
    for requested_layer in layers:
        hidden_layer = resolve_layer_index([None] * num_hidden_states, requested_layer)
        decoder_layer = hidden_layer - 1 if hidden_layer > 0 else 0
        direction, direction_tasks = build_component_direction(
            activation_summary=artifacts["activation_summary"],
            summary_vectors=artifacts["summary_vectors"],
            positive_prompt_id=args.positive_prompt_id,
            negative_prompt_id=args.negative_prompt_id,
            layer=hidden_layer,
            position=args.direction_position,
            tasks=config["tasks"]["seen"],
        )
        random_direction = matched_random_direction(direction, rng)
        orthogonal_direction = orthogonal_random_direction(direction, rng)
        metadata_rows.append(
            {
                "requested_layer": requested_layer,
                "hidden_state_layer": hidden_layer,
                "decoder_layer": decoder_layer,
                "direction_norm": float(np.linalg.norm(direction)),
                "direction_tasks": direction_tasks,
            }
        )

        for task_name, dataset in datasets.items():
            split = "seen" if task_name in config["tasks"]["seen"] else "unseen"
            baseline_specs = [
                ("target_baseline", target_prompt, None, 0.0, None, "none"),
                ("donor_baseline", positive_prompt, None, 0.0, None, "none"),
            ]
            for condition, prompt_record, vector, alpha, target_position, control_type in baseline_specs:
                for row in evaluate_condition(
                    model=model,
                    tokenizer=tokenizer,
                    prompt_record=prompt_record,
                    dataset=dataset,
                    task_name=task_name,
                    split=split,
                    layer=None,
                    vector=vector,
                    alpha=alpha,
                    target_position=target_position,
                    max_new_tokens=max_new_tokens,
                ):
                    row.update(
                        {
                            "condition": condition,
                            "layer": np.nan,
                            "hidden_state_layer": np.nan,
                            "direction_position": args.direction_position,
                            "target_position": target_position,
                            "alpha": np.nan,
                            "patch_prompt_id": prompt_record["id"],
                            "direction_prompt_id": f"{args.positive_prompt_id}-{args.negative_prompt_id}",
                            "control_type": control_type,
                        }
                    )
                    sample_rows.append(row)

            for target_position in args.target_positions:
                patch_specs = []
                for alpha in args.alphas:
                    patch_specs.append(("activation_patch", target_prompt, direction, alpha, "component_direction"))
                    patch_specs.append(("direction_removal", removal_prompt, direction, -alpha, "component_direction"))
                    if args.include_controls:
                        patch_specs.append(("random_matched_norm", target_prompt, random_direction, alpha, "random"))
                        patch_specs.append(("orthogonal_matched_norm", target_prompt, orthogonal_direction, alpha, "orthogonal_random"))

                for condition, prompt_record, vector, alpha, control_type in patch_specs:
                    for row in evaluate_condition(
                        model=model,
                        tokenizer=tokenizer,
                        prompt_record=prompt_record,
                        dataset=dataset,
                        task_name=task_name,
                        split=split,
                        layer=decoder_layer,
                        vector=vector,
                        alpha=alpha,
                        target_position=target_position,
                        max_new_tokens=max_new_tokens,
                    ):
                        row.update(
                            {
                                "condition": condition,
                                "layer": decoder_layer,
                                "hidden_state_layer": hidden_layer,
                                "direction_position": args.direction_position,
                                "target_position": target_position,
                                "alpha": alpha,
                                "patch_prompt_id": prompt_record["id"],
                                "direction_prompt_id": f"{args.positive_prompt_id}-{args.negative_prompt_id}",
                                "control_type": control_type,
                            }
                        )
                        sample_rows.append(row)

    sample_table = pd.DataFrame(sample_rows)
    summary_table = summarize(sample_rows)
    outputs_root = artifacts["outputs_root"]
    results_dir = ensure_dir(outputs_root / args.output_subdir)
    save_dataframe(sample_table, results_dir / "component_patching_samples.parquet")
    save_json(results_dir / "component_patching_samples.json", sample_table.to_dict(orient="records"))
    save_dataframe(summary_table, results_dir / "component_patching_summary.parquet")
    save_json(results_dir / "component_patching_summary.json", summary_table.to_dict(orient="records"))
    metadata = {
        "config": args.config,
        "positive_prompt_id": args.positive_prompt_id,
        "negative_prompt_id": args.negative_prompt_id,
        "target_prompt_id": args.target_prompt_id,
        "remove_from_prompt_id": args.remove_from_prompt_id,
        "direction_position": args.direction_position,
        "direction_tasks": sorted({task for row in metadata_rows for task in row["direction_tasks"]}),
        "layers": metadata_rows,
        "target_positions": args.target_positions,
        "alphas": args.alphas,
    }
    save_json(results_dir / "component_patching_metadata.json", metadata)
    save_markdown(results_dir / "component_patching_report.md", format_report(summary_table, metadata))
    print(f"Saved component patching outputs to {results_dir}")


if __name__ == "__main__":
    main()
