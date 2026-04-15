from __future__ import annotations

import argparse

import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.eval.evaluator import evaluate_dataset, summarize_prompt_metrics
from src.model.load_model import load_model
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
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt evaluation across tasks.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    set_seed(config.get("seed", 42))

    prompts_path = resolve_path(project_root, config["paths"]["prompts"])
    datasets_dir = resolve_path(project_root, config["paths"]["datasets_dir"])
    outputs_dir = ensure_dir(resolve_path(project_root, config["paths"]["outputs_dir"]) / "eval")

    prompts = ensure_base_prompt_record(
        load_prompts(prompts_path),
        base_prompt=config["base_prompt"],
    )
    if args.limit_prompts is not None:
        prompts = prompts[: args.limit_prompts]

    task_names = flatten_task_config(config)
    datasets = {}
    for task_name in task_names:
        dataset_path = datasets_dir / f"{task_name}.json"
        datasets[task_name] = load_dataset(
            dataset_path,
            task_name=task_name,
            limit=args.limit_per_task or config.get("limit_per_task"),
        )

    model, tokenizer = load_model(
        model_name=config["model_name"],
        torch_dtype=config.get("torch_dtype", "auto"),
        device_map=config.get("device_map", "auto"),
    )

    sample_rows = []
    total_prompts = len(prompts)
    for prompt_index, prompt_record in enumerate(prompts, start=1):
        print(f"[run_eval] prompt {prompt_index}/{total_prompts}: {prompt_record['id']}", flush=True)
        for task_name, dataset in datasets.items():
            split = "seen" if task_name in config["tasks"]["seen"] else "unseen"
            sample_rows.extend(
                evaluate_dataset(
                    model=model,
                    tokenizer=tokenizer,
                    dataset=dataset,
                    system_prompt=prompt_record["text"],
                    prompt_record=prompt_record,
                    task_name=task_name,
                    split=split,
                    batch_size=config["batch_size"],
                    max_new_tokens=config["max_new_tokens"],
                )
            )

    sample_df = pd.DataFrame(sample_rows)
    task_summary, prompt_summary = summarize_prompt_metrics(
        sample_rows=sample_rows,
        seen_tasks=config["tasks"]["seen"],
        unseen_tasks=config["tasks"]["unseen"],
    )

    save_dataframe(sample_df, outputs_dir / "eval_results.parquet")
    save_dataframe(task_summary, outputs_dir / "eval_task_summary.parquet")
    save_dataframe(prompt_summary, outputs_dir / "eval_prompt_summary.parquet")

    print(f"Saved {len(sample_df)} sample-level evaluation rows to {outputs_dir}")


if __name__ == "__main__":
    main()
