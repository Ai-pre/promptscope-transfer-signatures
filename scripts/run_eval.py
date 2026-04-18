from __future__ import annotations

import argparse
import json

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
    save_json,
    set_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run prompt evaluation across tasks.")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--limit-per-task", type=int, default=None)
    parser.add_argument("--limit-prompts", type=int, default=None)
    return parser.parse_args()


def build_prompt_metadata_frame(prompts):
    frame = pd.DataFrame(prompts).rename(columns={"id": "prompt_id"})
    meta_columns = [
        "prompt_id",
        "group_id",
        "variant",
        "source",
        "prompt_role",
        "original_prompt_role",
        "task_scope",
        "provenance",
        "source_title",
        "source_url",
        "paper_title",
        "paper_url",
        "source_note",
        "optimized_for_tasks_json",
        "source_datasets_json",
        "is_paper_backed",
        "principle_family",
        "principle_components_json",
        "complexity_level",
        "hypothesis_role",
        "contrast_group",
    ]
    available = [column for column in meta_columns if column in frame.columns]
    return frame[available].drop_duplicates(subset=["prompt_id"]).reset_index(drop=True)


def parse_json_task_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed if str(item).strip()]
        return [stripped]
    return [str(value).strip()]


def build_seen_fit_table(task_summary, prompt_summary, seen_tasks):
    seen_task_df = task_summary[task_summary["task"].isin(seen_tasks)].copy()
    if seen_task_df.empty:
        return prompt_summary.copy()

    seen_stats = (
        seen_task_df.groupby("prompt_id", dropna=False)["accuracy"]
        .agg(seen_min_accuracy="min", seen_std_accuracy="std", seen_task_count="count")
        .reset_index()
    )
    seen_fit = prompt_summary.merge(seen_stats, on="prompt_id", how="left")
    seen_fit["seen_std_accuracy"] = seen_fit["seen_std_accuracy"].fillna(0.0)
    seen_fit["seen_task_count"] = seen_fit["seen_task_count"].fillna(0).astype(int)

    seen_task_set = set(seen_tasks)
    overlaps = []
    overlap_labels = []
    alignment_flags = []
    for row in seen_fit.itertuples(index=False):
        optimized_tasks = parse_json_task_list(getattr(row, "optimized_for_tasks_json", "[]"))
        overlap = sorted(seen_task_set.intersection(optimized_tasks))
        overlaps.append(len(overlap))
        overlap_labels.append(json.dumps(overlap, ensure_ascii=False))
        alignment_flags.append(getattr(row, "task_scope", "") == "task_agnostic" or bool(overlap))

    seen_fit["seen_task_overlap_count"] = overlaps
    seen_fit["seen_task_overlap_json"] = overlap_labels
    seen_fit["is_seen_task_aligned"] = alignment_flags
    return rank_seen_fit_table(seen_fit)


def rank_seen_fit_table(frame: pd.DataFrame):
    ranked = frame.sort_values(
        ["is_seen_task_aligned", "seen_mean_accuracy", "seen_min_accuracy", "overall_accuracy"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    ranked["seen_fit_rank"] = ranked.index + 1
    return ranked


def build_original_seen_fit_table(seen_fit_summary: pd.DataFrame):
    originals = seen_fit_summary[seen_fit_summary["variant"] == "original"].copy()
    if originals.empty:
        return originals
    return rank_seen_fit_table(originals)


def build_group_seen_fit_table(seen_fit_summary: pd.DataFrame):
    if seen_fit_summary.empty:
        return seen_fit_summary.copy()

    rows = []
    for group_id, group in seen_fit_summary.groupby("group_id", dropna=False):
        original_rows = group[group["variant"] == "original"]
        representative = original_rows.iloc[0] if not original_rows.empty else group.iloc[0]
        rows.append(
            {
                "group_id": group_id,
                "representative_prompt_id": representative["prompt_id"],
                "source": representative["source"],
                "source_title": representative.get("source_title"),
                "source_url": representative.get("source_url"),
                "paper_title": representative.get("paper_title"),
                "paper_url": representative.get("paper_url"),
                "provenance": representative.get("provenance"),
                "task_scope": representative.get("task_scope"),
                "prompt_role": representative.get("prompt_role"),
                "original_prompt_role": representative.get("original_prompt_role"),
                "optimized_for_tasks_json": representative.get("optimized_for_tasks_json"),
                "source_datasets_json": representative.get("source_datasets_json"),
                "is_paper_backed": bool(representative.get("is_paper_backed", False)),
                "num_variants": int(len(group)),
                "variants_json": json.dumps(sorted(group["variant"].astype(str).unique().tolist()), ensure_ascii=False),
                "prompt_ids_json": json.dumps(group["prompt_id"].astype(str).tolist(), ensure_ascii=False),
                "seen_mean_accuracy": float(group["seen_mean_accuracy"].mean()),
                "unseen_mean_accuracy": float(group["unseen_mean_accuracy"].mean()),
                "overall_accuracy": float(group["overall_accuracy"].mean()),
                "transfer_gap": float(group["transfer_gap"].mean()),
                "sensitivity": float(group["sensitivity"].mean()),
                "seen_min_accuracy": float(group["seen_min_accuracy"].min()),
                "seen_std_accuracy": float(group["seen_std_accuracy"].mean()),
                "seen_task_count": int(group["seen_task_count"].max()),
                "seen_task_overlap_count": int(group["seen_task_overlap_count"].max()),
                "seen_task_overlap_json": representative.get("seen_task_overlap_json", "[]"),
                "is_seen_task_aligned": bool(group["is_seen_task_aligned"].all()),
            }
        )

    return rank_seen_fit_table(pd.DataFrame(rows))


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
    prompt_limit = args.limit_prompts if args.limit_prompts is not None else config.get("limit_prompts")
    if prompt_limit is not None:
        prompts = prompts[: int(prompt_limit)]

    task_names = flatten_task_config(config)
    datasets = {}
    for task_name in task_names:
        dataset_path = datasets_dir / f"{task_name}.json"
        datasets[task_name] = load_dataset(
            dataset_path,
            task_name=task_name,
            limit=args.limit_per_task if args.limit_per_task is not None else config.get("limit_per_task"),
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
            print(
                f"[run_eval]   task={task_name} split={split} samples={len(dataset)}",
                flush=True,
            )
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
    prompt_metadata = build_prompt_metadata_frame(prompts)
    task_summary = task_summary.merge(prompt_metadata, on=["prompt_id", "group_id", "variant", "source"], how="left")
    prompt_summary = prompt_summary.merge(prompt_metadata, on=["prompt_id", "group_id", "variant", "source"], how="left")
    seen_fit_summary = build_seen_fit_table(
        task_summary=task_summary,
        prompt_summary=prompt_summary,
        seen_tasks=config["tasks"]["seen"],
    )
    seen_fit_originals = build_original_seen_fit_table(seen_fit_summary)
    seen_fit_groups = build_group_seen_fit_table(seen_fit_summary)

    save_dataframe(sample_df, outputs_dir / "eval_results.parquet")
    save_dataframe(task_summary, outputs_dir / "eval_task_summary.parquet")
    save_dataframe(prompt_summary, outputs_dir / "eval_prompt_summary.parquet")
    save_dataframe(seen_fit_summary, outputs_dir / "eval_seen_fit_summary.parquet")
    save_json(outputs_dir / "eval_seen_fit_summary.json", seen_fit_summary.to_dict(orient="records"))
    save_dataframe(seen_fit_originals, outputs_dir / "eval_seen_fit_originals.parquet")
    save_json(outputs_dir / "eval_seen_fit_originals.json", seen_fit_originals.to_dict(orient="records"))
    save_dataframe(seen_fit_groups, outputs_dir / "eval_seen_fit_groups.parquet")
    save_json(outputs_dir / "eval_seen_fit_groups.json", seen_fit_groups.to_dict(orient="records"))

    print(f"Saved {len(sample_df)} sample-level evaluation rows to {outputs_dir}")


if __name__ == "__main__":
    main()
