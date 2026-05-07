from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path, save_dataframe, save_json, save_markdown


def parse_component_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze controlled length-vs-boundary prompts after run_eval has "
            "produced eval_results.parquet."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="length_control_results")
    return parser.parse_args()


def load_prompt_metadata(config):
    prompts_path = resolve_path(config["_project_root"], config["paths"]["prompts"])
    prompt_meta = pd.DataFrame(load_prompts(prompts_path)).rename(columns={"id": "prompt_id"})
    keep = [
        "prompt_id",
        "group_id",
        "source",
        "text",
        "principle_components_json",
        "length_control_role",
        "source_note",
    ]
    available = [column for column in keep if column in prompt_meta.columns]
    prompt_meta = prompt_meta[available].copy()
    if "principle_components_json" in prompt_meta.columns:
        prompt_meta["principle_components"] = prompt_meta["principle_components_json"].apply(parse_component_list)
    else:
        prompt_meta["principle_components"] = [[] for _ in range(len(prompt_meta))]
        prompt_meta["principle_components_json"] = "[]"
    if "length_control_role" not in prompt_meta.columns:
        prompt_meta["length_control_role"] = "unknown"
    return prompt_meta


def count_words(text):
    return len(str(text).strip().split())


def has_final_answer_marker(text):
    return "final answer" in str(text).lower()


def build_prompt_table(eval_results, prompt_meta):
    frame = eval_results.copy()
    frame["prediction_word_count"] = frame["prediction"].apply(count_words)
    frame["has_final_answer_marker"] = frame["prediction"].apply(has_final_answer_marker)

    grouped = (
        frame.groupby(["prompt_id", "group_id", "source", "task", "split"], dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_prediction_words=("prediction_word_count", "mean"),
            median_prediction_words=("prediction_word_count", "median"),
            final_answer_marker_rate=("has_final_answer_marker", "mean"),
            num_samples=("correct", "count"),
        )
        .reset_index()
    )

    prompt_rows = []
    for key, group in grouped.groupby(["prompt_id", "group_id", "source"], dropna=False):
        row = {column: value for column, value in zip(["prompt_id", "group_id", "source"], key)}
        seen = group[group["split"] == "seen"]
        unseen = group[group["split"] == "unseen"]
        row["seen_mean_accuracy"] = float(seen["accuracy"].mean()) if not seen.empty else float("nan")
        row["unseen_mean_accuracy"] = float(unseen["accuracy"].mean()) if not unseen.empty else float("nan")
        row["overall_accuracy"] = float(group["accuracy"].mean())
        row["mean_prediction_words"] = float(group["mean_prediction_words"].mean())
        row["median_prediction_words"] = float(group["median_prediction_words"].median())
        row["final_answer_marker_rate"] = float(group["final_answer_marker_rate"].mean())
        prompt_rows.append(row)

    prompt_table = pd.DataFrame(prompt_rows)
    prompt_table = prompt_table.merge(
        prompt_meta,
        on=["prompt_id", "group_id", "source"],
        how="left",
    )
    return grouped, prompt_table


def component_effects(prompt_table):
    rows = []
    components = sorted(
        {
            component
            for components in prompt_table["principle_components"]
            for component in components
        }
    )
    for component in components:
        present = prompt_table[prompt_table["principle_components"].apply(lambda items: component in items)]
        absent = prompt_table[prompt_table["principle_components"].apply(lambda items: component not in items)]
        if present.empty or absent.empty:
            continue
        rows.append(
            {
                "component": component,
                "present_count": int(len(present)),
                "absent_count": int(len(absent)),
                "delta_unseen_mean_accuracy": float(
                    present["unseen_mean_accuracy"].mean() - absent["unseen_mean_accuracy"].mean()
                ),
                "delta_seen_mean_accuracy": float(
                    present["seen_mean_accuracy"].mean() - absent["seen_mean_accuracy"].mean()
                ),
                "delta_mean_prediction_words": float(
                    present["mean_prediction_words"].mean() - absent["mean_prediction_words"].mean()
                ),
                "delta_final_answer_marker_rate": float(
                    present["final_answer_marker_rate"].mean() - absent["final_answer_marker_rate"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def role_effects(prompt_table):
    role_table = (
        prompt_table.groupby("length_control_role", dropna=False)
        .agg(
            prompt_count=("prompt_id", "count"),
            seen_mean_accuracy=("seen_mean_accuracy", "mean"),
            unseen_mean_accuracy=("unseen_mean_accuracy", "mean"),
            overall_accuracy=("overall_accuracy", "mean"),
            mean_prediction_words=("mean_prediction_words", "mean"),
            final_answer_marker_rate=("final_answer_marker_rate", "mean"),
        )
        .reset_index()
        .sort_values(["unseen_mean_accuracy", "overall_accuracy"], ascending=[False, False])
    )
    return role_table


def safe_corr(left, right):
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def build_summary(prompt_table, role_table, component_table):
    best_prompt = prompt_table.sort_values(
        ["unseen_mean_accuracy", "overall_accuracy"],
        ascending=[False, False],
    ).iloc[0]
    return {
        "num_prompts": int(len(prompt_table)),
        "best_unseen_prompt": str(best_prompt["prompt_id"]),
        "best_unseen_accuracy": float(best_prompt["unseen_mean_accuracy"]),
        "best_unseen_role": str(best_prompt.get("length_control_role", "")),
        "prediction_length_pearson_unseen": safe_corr(
            prompt_table["mean_prediction_words"],
            prompt_table["unseen_mean_accuracy"],
        ),
        "final_answer_marker_pearson_unseen": safe_corr(
            prompt_table["final_answer_marker_rate"],
            prompt_table["unseen_mean_accuracy"],
        ),
        "role_effects": role_table.to_dict(orient="records"),
        "component_effects": component_table.to_dict(orient="records"),
    }


def format_report(summary, prompt_table, role_table, component_table):
    lines = [
        "# Length-Control Disentanglement Report",
        "",
        "## Summary",
        "",
        f"- Best unseen prompt: {summary['best_unseen_prompt']}",
        f"- Best unseen accuracy: {summary['best_unseen_accuracy']:.4f}",
        f"- Best unseen role: {summary['best_unseen_role']}",
        f"- Prediction length Pearson with unseen: {summary['prediction_length_pearson_unseen']:.4f}",
        f"- FINAL ANSWER marker Pearson with unseen: {summary['final_answer_marker_pearson_unseen']:.4f}",
        "",
        "## Role Effects",
        "",
    ]
    for row in role_table.itertuples(index=False):
        lines.append(
            f"- {row.length_control_role}: unseen={row.unseen_mean_accuracy:.4f}, "
            f"seen={row.seen_mean_accuracy:.4f}, words={row.mean_prediction_words:.2f}, "
            f"marker={row.final_answer_marker_rate:.3f}"
        )
    lines.extend(["", "## Component Effects", ""])
    for row in component_table.itertuples(index=False):
        lines.append(
            f"- {row.component}: delta_unseen={row.delta_unseen_mean_accuracy:.4f}, "
            f"delta_words={row.delta_mean_prediction_words:.2f}, "
            f"delta_marker={row.delta_final_answer_marker_rate:.3f}"
        )
    lines.extend(["", "## Prompt Ranking", ""])
    ranked = prompt_table.sort_values(["unseen_mean_accuracy", "overall_accuracy"], ascending=[False, False])
    for row in ranked.itertuples(index=False):
        lines.append(
            f"- {row.prompt_id}: role={row.length_control_role}, unseen={row.unseen_mean_accuracy:.4f}, "
            f"seen={row.seen_mean_accuracy:.4f}, words={row.mean_prediction_words:.2f}, "
            f"marker={row.final_answer_marker_rate:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)
    outputs_root = resolve_path(config["_project_root"], config["paths"]["outputs_dir"])
    eval_dir = outputs_root / "eval"
    results_dir = ensure_dir(outputs_root / args.output_subdir)

    eval_results = pd.read_parquet(eval_dir / "eval_results.parquet")
    prompt_meta = load_prompt_metadata(config)
    task_table, prompt_table = build_prompt_table(eval_results, prompt_meta)
    component_table = component_effects(prompt_table)
    role_table = role_effects(prompt_table)
    summary = build_summary(prompt_table, role_table, component_table)
    summary.update(
        {
            "config": args.config,
            "outputs_dir": str(outputs_root),
        }
    )

    save_dataframe(task_table, results_dir / "length_control_task_table.parquet")
    save_json(results_dir / "length_control_task_table.json", task_table.to_dict(orient="records"))
    save_dataframe(prompt_table, results_dir / "length_control_prompt_table.parquet")
    save_json(results_dir / "length_control_prompt_table.json", prompt_table.to_dict(orient="records"))
    save_dataframe(role_table, results_dir / "length_control_role_effects.parquet")
    save_json(results_dir / "length_control_role_effects.json", role_table.to_dict(orient="records"))
    save_dataframe(component_table, results_dir / "length_control_component_effects.parquet")
    save_json(results_dir / "length_control_component_effects.json", component_table.to_dict(orient="records"))
    save_json(results_dir / "length_control_summary.json", summary)
    save_markdown(
        results_dir / "length_control_report.md",
        format_report(summary, prompt_table, role_table, component_table),
    )
    print(f"Saved length-control analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()
