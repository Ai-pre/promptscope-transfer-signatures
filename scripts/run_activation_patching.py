from __future__ import annotations

import argparse
from contextlib import contextmanager

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.activation.extractor import extract_hidden_states, get_positions, select_hidden_vector
from src.eval.evaluator import compute_accuracy, generate
from src.model.load_model import get_model_device, load_model
from src.prompt.prompt_builder import build_input
from src.utils.io import (
    ensure_dir,
    flatten_task_config,
    load_config,
    load_dataset,
    load_prompts,
    resolve_path,
    save_dataframe,
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run causal activation patching from a donor prompt into a target prompt."
    )
    parser.add_argument("--config", default="configs/config.gpu_activation_patch.yaml")
    parser.add_argument("--donor-prompt-id", default=None)
    parser.add_argument("--target-prompt-id", default=None)
    parser.add_argument("--limit-per-task", type=int, default=None)
    return parser.parse_args()


def get_by_prompt_id(prompts, prompt_id: str):
    for prompt in prompts:
        if prompt["id"] == prompt_id:
            return prompt
    raise ValueError(f"Prompt id not found: {prompt_id}")


def get_transformer_layers(model):
    for root_name in ("model", "transformer", "gpt_neox"):
        root = getattr(model, root_name, None)
        if root is None:
            continue
        layers = getattr(root, "layers", None)
        if layers is not None:
            return layers
        layers = getattr(root, "h", None)
        if layers is not None:
            return layers
    raise ValueError("Could not locate transformer layers for activation patching.")


def hidden_state_layer_to_module_index(hidden_state_layer: int):
    if hidden_state_layer <= 0:
        raise ValueError(
            "Cannot patch hidden_states[0] because it is the embedding state, "
            "not a transformer block output."
        )
    return hidden_state_layer - 1


@contextmanager
def patch_layer_output(model, *, hidden_state_layer: int, token_index: int, donor_vector, alpha: float):
    layers = get_transformer_layers(model)
    module_index = hidden_state_layer_to_module_index(hidden_state_layer)
    if module_index < 0 or module_index >= len(layers):
        raise IndexError(
            f"Layer {hidden_state_layer} maps to module {module_index}, "
            f"but the model has {len(layers)} transformer blocks."
        )

    state = {"patched": False}

    def hook(_module, _inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        if state["patched"] or hidden.ndim != 3 or hidden.shape[1] <= token_index:
            return output

        patched_hidden = hidden.clone()
        donor = donor_vector.to(device=patched_hidden.device, dtype=patched_hidden.dtype)
        current = patched_hidden[:, token_index, :]
        patched_hidden[:, token_index, :] = (1.0 - alpha) * current + alpha * donor.unsqueeze(0)
        state["patched"] = True

        if isinstance(output, tuple):
            return (patched_hidden,) + output[1:]
        return patched_hidden

    handle = layers[module_index].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def render_and_positions(tokenizer, prompt_text: str, user_input: str):
    rendered = build_input(tokenizer, prompt_text, user_input)
    positions = get_positions(
        tokenizer=tokenizer,
        rendered_prompt=rendered,
        system_prompt=prompt_text,
        user_input=user_input,
    )
    return rendered, positions


def evaluate_prediction(prediction: str, sample: dict, task_name: str):
    correct, pred_norm, gold_norm = compute_accuracy(
        prediction=prediction,
        gold_label=sample["label"],
        task_name=task_name,
        sample=sample,
    )
    return {
        "prediction": prediction,
        "normalized_prediction": pred_norm,
        "normalized_label": gold_norm,
        "correct": correct,
    }


def resolve_patch_tasks(config):
    patching = config.get("patching", {})
    if "tasks" in patching:
        return list(patching["tasks"])
    task_split = patching.get("task_split", "unseen")
    if task_split == "seen":
        return list(config["tasks"]["seen"])
    if task_split == "unseen":
        return list(config["tasks"]["unseen"])
    if task_split == "all":
        return flatten_task_config(config)
    raise ValueError(f"Unsupported patching.task_split={task_split!r}")


def summarize_rows(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    group_columns = ["condition", "task", "layer", "position", "alpha"]
    return (
        frame.groupby(group_columns, dropna=False)["correct"]
        .agg(accuracy="mean", num_samples="count")
        .reset_index()
    )


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    set_seed(config.get("seed", 42))

    patching = config.get("patching", {})
    donor_prompt_id = args.donor_prompt_id or patching.get("donor_prompt_id")
    target_prompt_id = args.target_prompt_id or patching.get("target_prompt_id")
    if not donor_prompt_id or not target_prompt_id:
        raise ValueError("Both donor and target prompt ids must be provided.")

    prompts = load_prompts(resolve_path(project_root, config["paths"]["prompts"]))
    donor_prompt = get_by_prompt_id(prompts, donor_prompt_id)
    target_prompt = get_by_prompt_id(prompts, target_prompt_id)

    datasets_dir = resolve_path(project_root, config["paths"]["datasets_dir"])
    task_names = resolve_patch_tasks(config)
    sample_limit = args.limit_per_task if args.limit_per_task is not None else patching.get("limit_per_task", config.get("limit_per_task"))
    datasets = {
        task_name: load_dataset(
            datasets_dir / f"{task_name}.json",
            task_name=task_name,
            limit=sample_limit,
        )
        for task_name in task_names
    }

    model, tokenizer = load_model(
        model_name=config["model_name"],
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=config.get("device_map", "auto"),
    )
    device = get_model_device(model)

    layers = patching.get("layers", config["layers"])
    positions = patching.get("positions", config["positions"])
    alphas = [float(value) for value in patching.get("alphas", [1.0])]

    rows = []
    for task_name, dataset in datasets.items():
        split = "seen" if task_name in config["tasks"]["seen"] else "unseen"
        print(f"[activation_patching] task={task_name} split={split} samples={len(dataset)}", flush=True)
        for sample_index, sample in enumerate(dataset, start=1):
            if sample_index % 10 == 0:
                print(f"[activation_patching]   sample {sample_index}/{len(dataset)}", flush=True)

            donor_text, donor_positions = render_and_positions(tokenizer, donor_prompt["text"], sample["input"])
            target_text, target_positions = render_and_positions(tokenizer, target_prompt["text"], sample["input"])

            donor_prediction = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=donor_text,
                max_new_tokens=config["max_new_tokens"],
            )
            target_prediction = generate(
                model=model,
                tokenizer=tokenizer,
                prompt=target_text,
                max_new_tokens=config["max_new_tokens"],
            )

            for condition, prompt_record, prediction in (
                ("donor_baseline", donor_prompt, donor_prediction),
                ("target_baseline", target_prompt, target_prediction),
            ):
                metrics = evaluate_prediction(prediction, sample, task_name)
                rows.append(
                    {
                        "condition": condition,
                        "prompt_id": prompt_record["id"],
                        "donor_prompt_id": donor_prompt["id"],
                        "target_prompt_id": target_prompt["id"],
                        "task": task_name,
                        "split": split,
                        "sample_id": sample["id"],
                        "layer": None,
                        "position": None,
                        "alpha": None,
                        **metrics,
                    }
                )

            donor_hidden, _ = extract_hidden_states(model, tokenizer, donor_text)
            for layer_spec in layers:
                for position_name in positions:
                    donor_layer, donor_vector = select_hidden_vector(
                        hidden_states=donor_hidden,
                        layer_spec=int(layer_spec),
                        token_index=donor_positions[position_name],
                    )
                    token_index = target_positions[position_name]

                    for alpha in alphas:
                        with patch_layer_output(
                            model,
                            hidden_state_layer=donor_layer,
                            token_index=token_index,
                            donor_vector=donor_vector.detach().to(device),
                            alpha=alpha,
                        ):
                            patched_prediction = generate(
                                model=model,
                                tokenizer=tokenizer,
                                prompt=target_text,
                                max_new_tokens=config["max_new_tokens"],
                            )

                        metrics = evaluate_prediction(patched_prediction, sample, task_name)
                        rows.append(
                            {
                                "condition": "activation_patch",
                                "prompt_id": target_prompt["id"],
                                "donor_prompt_id": donor_prompt["id"],
                                "target_prompt_id": target_prompt["id"],
                                "task": task_name,
                                "split": split,
                                "sample_id": sample["id"],
                                "layer": donor_layer,
                                "position": position_name,
                                "alpha": alpha,
                                **metrics,
                            }
                        )

            del donor_hidden

    outputs_dir = ensure_dir(resolve_path(project_root, config["paths"]["outputs_dir"]) / "activation_patching")
    sample_frame = pd.DataFrame(rows)
    summary_frame = summarize_rows(rows)
    save_dataframe(sample_frame, outputs_dir / "activation_patching_samples.parquet")
    save_dataframe(summary_frame, outputs_dir / "activation_patching_summary.parquet")
    save_json(outputs_dir / "activation_patching_samples.json", sample_frame.to_dict(orient="records"))
    save_json(outputs_dir / "activation_patching_summary.json", summary_frame.to_dict(orient="records"))

    print(f"Saved activation patching outputs to {outputs_dir}", flush=True)


if __name__ == "__main__":
    main()
