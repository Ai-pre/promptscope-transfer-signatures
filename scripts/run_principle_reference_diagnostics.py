from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from run_principle_analysis import (
    build_slice_feature_table,
    load_run_artifacts,
    select_reference_slice,
)
from src.analysis.analyzer import cosine, lexical_similarity
from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose principle prompts against paper-backed reference prompts using "
            "nearest-neighbor and family-centroid activation similarities."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--nearest-neighbors", type=int, default=5)
    parser.add_argument("--top-reference-prompts", type=int, default=10)
    return parser.parse_args()


def prompt_family(row):
    prompt_id = str(row.get("prompt_id", ""))
    source = str(row.get("source", ""))

    if "emotion_prompt" in prompt_id:
        return "emotion_prompt"
    if "generated_knowledge" in prompt_id:
        return "generated_knowledge"
    if "least_to_most" in prompt_id:
        return "least_to_most"
    if "zscot" in prompt_id or "zero_shot_cot" in prompt_id:
        return "zero_shot_cot"
    if "_cot_" in prompt_id or prompt_id.endswith("_cot"):
        return "few_shot_cot"
    if "answer_only" in prompt_id:
        return "answer_only"
    if "direct_letter" in prompt_id:
        return "direct_letter"
    if "baseline" in prompt_id:
        return "baseline"
    if "promptwizard" in prompt_id or source == "promptwizard":
        return "promptwizard"
    if "meta_prompting" in prompt_id or source == "meta_prompting":
        return "meta_prompting"
    if source:
        return source
    return "unknown"


def merge_prompt_metadata(table, prompts):
    prompt_meta = pd.DataFrame(prompts).rename(columns={"id": "prompt_id"})
    keep_columns = [
        column
        for column in [
            "prompt_id",
            "source_title",
            "source_url",
            "paper_title",
            "paper_url",
            "provenance",
            "task_scope",
            "optimized_for_tasks_json",
            "source_datasets_json",
            "source_note",
        ]
        if column in prompt_meta.columns
    ]
    return table.merge(prompt_meta[keep_columns], on="prompt_id", how="left", suffixes=("", "_meta"))


def finite_float(value):
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def rank_reference_prompts(principle_row, principle_vector, reference_table, reference_features, top_n):
    rows = []
    for reference_index, reference_row in reference_table.reset_index(drop=True).iterrows():
        similarity = cosine(principle_vector, reference_features[reference_index])
        rows.append(
            {
                "principle_prompt_id": principle_row["prompt_id"],
                "reference_prompt_id": reference_row["prompt_id"],
                "reference_family": reference_row["reference_family"],
                "reference_source": reference_row.get("source"),
                "cosine_similarity": similarity,
                "lexical_similarity": lexical_similarity(
                    str(principle_row.get("prompt_text", "")),
                    str(reference_row.get("prompt_text", "")),
                ),
                "reference_seen_mean_accuracy": finite_float(reference_row.get("seen_mean_accuracy")),
                "reference_unseen_mean_accuracy": finite_float(reference_row.get("unseen_mean_accuracy")),
                "reference_overall_accuracy": finite_float(reference_row.get("overall_accuracy")),
                "reference_prompt_length_words": finite_float(reference_row.get("prompt_length_words")),
                "reference_paper_title": reference_row.get("paper_title"),
                "reference_source_title": reference_row.get("source_title"),
            }
        )
    return sorted(rows, key=lambda row: row["cosine_similarity"], reverse=True)[:top_n]


def build_nearest_neighbor_table(principle_table, principle_features, reference_table, reference_features, top_n):
    rows = []
    for principle_index, principle_row in principle_table.reset_index(drop=True).iterrows():
        rows.extend(
            rank_reference_prompts(
                principle_row=principle_row,
                principle_vector=principle_features[principle_index],
                reference_table=reference_table,
                reference_features=reference_features,
                top_n=top_n,
            )
        )
    return pd.DataFrame(rows)


def build_family_centroid_table(principle_table, principle_features, reference_table, reference_features, *, top_reference_ids=None):
    reference = reference_table.reset_index(drop=True).copy()
    features = reference_features
    if top_reference_ids is not None:
        mask = reference["prompt_id"].isin(top_reference_ids).to_numpy()
        reference = reference[mask].reset_index(drop=True)
        features = features[mask]

    rows = []
    for family, family_table in reference.groupby("reference_family", dropna=False):
        indices = family_table.index.to_numpy()
        centroid = np.mean(features[indices], axis=0)
        best_family_prompt = family_table.sort_values(
            ["unseen_mean_accuracy", "overall_accuracy", "seen_mean_accuracy"],
            ascending=[False, False, False],
        ).iloc[0]
        for principle_index, principle_row in principle_table.reset_index(drop=True).iterrows():
            rows.append(
                {
                    "principle_prompt_id": principle_row["prompt_id"],
                    "reference_family": family,
                    "family_size": int(len(family_table)),
                    "family_centroid_cosine": cosine(principle_features[principle_index], centroid),
                    "family_mean_unseen_accuracy": finite_float(family_table["unseen_mean_accuracy"].mean()),
                    "family_mean_seen_accuracy": finite_float(family_table["seen_mean_accuracy"].mean()),
                    "family_best_prompt_id": best_family_prompt["prompt_id"],
                    "family_best_unseen_accuracy": finite_float(best_family_prompt["unseen_mean_accuracy"]),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["principle_prompt_id", "family_centroid_cosine"],
        ascending=[True, False],
    )


def best_rows_by_prompt(table, score_column):
    if table.empty:
        return []
    rows = []
    for _, group in table.groupby("principle_prompt_id", dropna=False):
        rows.append(group.sort_values(score_column, ascending=False).iloc[0].to_dict())
    return rows


def format_report(summary, nearest_table, top_nearest_table, family_table, top_family_table):
    lines = [
        "# Principle Reference Diagnostics",
        "",
        "## Method",
        "",
        "- Uses the same selected reference slice as `run_principle_analysis.py`.",
        "- Replaces a single top-prompt centroid with two less-collapsed diagnostics.",
        "- Nearest-neighbor: compares each principle prompt to individual paper-backed reference prompts.",
        "- Family centroid: groups reference prompts by inferred prompt family/source before averaging.",
        "",
        "## Summary",
        "",
        f"- Reference slice: layer={summary['reference_slice']['layer']}, position={summary['reference_slice']['position']}",
        f"- Reference prompts: {summary['num_reference_prompts']}",
        f"- Principle prompts: {summary['num_principle_prompts']}",
        "",
        "## Best Individual Reference Neighbor",
        "",
    ]

    for row in best_rows_by_prompt(nearest_table, "cosine_similarity"):
        lines.append(
            f"- {row['principle_prompt_id']} -> {row['reference_prompt_id']} "
            f"({row['reference_family']}), cosine={row['cosine_similarity']:.4f}, "
            f"ref_unseen={row['reference_unseen_mean_accuracy']:.4f}"
        )

    if not top_nearest_table.empty:
        lines.extend(["", "## Best Top-Reference Neighbor", ""])
        for row in best_rows_by_prompt(top_nearest_table, "cosine_similarity"):
            lines.append(
                f"- {row['principle_prompt_id']} -> {row['reference_prompt_id']} "
                f"({row['reference_family']}), cosine={row['cosine_similarity']:.4f}, "
                f"ref_unseen={row['reference_unseen_mean_accuracy']:.4f}"
            )

    if not family_table.empty:
        lines.extend(["", "## Best Reference Family", ""])
        for row in best_rows_by_prompt(family_table, "family_centroid_cosine"):
            lines.append(
                f"- {row['principle_prompt_id']} -> {row['reference_family']}, "
                f"cosine={row['family_centroid_cosine']:.4f}, "
                f"family_mean_unseen={row['family_mean_unseen_accuracy']:.4f}"
            )

    if not top_family_table.empty:
        lines.extend(["", "## Best Top-Reference Family", ""])
        for row in best_rows_by_prompt(top_family_table, "family_centroid_cosine"):
            lines.append(
                f"- {row['principle_prompt_id']} -> {row['reference_family']}, "
                f"cosine={row['family_centroid_cosine']:.4f}, "
                f"family_mean_unseen={row['family_mean_unseen_accuracy']:.4f}"
            )

    lines.extend(
        [
            "",
            "## Interpretation Note",
            "",
            "These diagnostics should not be read as causal directions. They are interpretability probes for whether controlled principle prompts sit near individual or family-specific paper-backed reference prompts in the selected activation space.",
        ]
    )
    return "\n".join(lines)


def main():
    args = parse_args()
    principle_config = load_config(args.config)
    reference_config = load_config(args.reference_config)

    principle_artifacts = load_run_artifacts(principle_config)
    reference_artifacts = load_run_artifacts(reference_config)
    reference_slice = select_reference_slice(reference_artifacts["slice_analysis"])
    if reference_slice is None:
        raise ValueError("Reference slice analysis is empty; run reference analysis first.")

    slice_layer = reference_slice.get("layer")
    slice_position = reference_slice.get("position")

    principle_features, _, principle_table, _ = build_slice_feature_table(
        principle_artifacts["activation_summary"],
        principle_artifacts["summary_vectors"],
        principle_artifacts["eval_prompt_summary"],
        principle_config["tasks"]["seen"],
        layer=slice_layer,
        position=slice_position,
    )
    principle_prompts = load_prompts(resolve_path(principle_config["_project_root"], principle_config["paths"]["prompts"]))
    principle_prompt_ids = {prompt["id"] for prompt in principle_prompts}
    principle_mask = principle_table["prompt_id"].isin(principle_prompt_ids).to_numpy()
    principle_features = principle_features[principle_mask]
    principle_table = principle_table[principle_mask].reset_index(drop=True)

    reference_features, _, reference_table, _ = build_slice_feature_table(
        reference_artifacts["activation_summary"],
        reference_artifacts["summary_vectors"],
        reference_artifacts["eval_prompt_summary"],
        reference_config["tasks"]["seen"],
        layer=slice_layer,
        position=slice_position,
    )
    reference_prompts = load_prompts(resolve_path(reference_config["_project_root"], reference_config["paths"]["prompts"]))
    reference_table = merge_prompt_metadata(reference_table, reference_prompts)
    reference_mask = (reference_table["source"] != "base").to_numpy()
    reference_features = reference_features[reference_mask]
    reference_table = reference_table[reference_mask].reset_index(drop=True)
    reference_table["reference_family"] = reference_table.apply(prompt_family, axis=1)

    ranked_reference = reference_table.sort_values(
        ["unseen_mean_accuracy", "overall_accuracy", "seen_mean_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top_reference_ids = ranked_reference.head(args.top_reference_prompts)["prompt_id"].tolist()
    top_reference_mask = reference_table["prompt_id"].isin(top_reference_ids).to_numpy()
    top_reference_table = reference_table[top_reference_mask].reset_index(drop=True)
    top_reference_features = reference_features[top_reference_mask]

    nearest_table = build_nearest_neighbor_table(
        principle_table,
        principle_features,
        reference_table,
        reference_features,
        args.nearest_neighbors,
    )
    top_nearest_table = build_nearest_neighbor_table(
        principle_table,
        principle_features,
        top_reference_table,
        top_reference_features,
        args.nearest_neighbors,
    )
    family_table = build_family_centroid_table(principle_table, principle_features, reference_table, reference_features)
    top_family_table = build_family_centroid_table(
        principle_table,
        principle_features,
        reference_table,
        reference_features,
        top_reference_ids=top_reference_ids,
    )

    summary = {
        "reference_slice": reference_slice,
        "num_reference_prompts": int(len(reference_table)),
        "num_principle_prompts": int(len(principle_table)),
        "nearest_neighbors_per_prompt": int(args.nearest_neighbors),
        "top_reference_prompts": top_reference_ids,
        "best_individual_neighbor_by_prompt": best_rows_by_prompt(nearest_table, "cosine_similarity"),
        "best_top_reference_neighbor_by_prompt": best_rows_by_prompt(top_nearest_table, "cosine_similarity"),
        "best_family_by_prompt": best_rows_by_prompt(family_table, "family_centroid_cosine"),
        "best_top_family_by_prompt": best_rows_by_prompt(top_family_table, "family_centroid_cosine"),
    }

    outputs_root = resolve_path(principle_config["_project_root"], principle_config["paths"]["outputs_dir"])
    results_dir = ensure_dir(outputs_root / "principle_reference_diagnostics")
    save_dataframe(nearest_table, results_dir / "principle_reference_nearest_neighbors.parquet")
    save_json(results_dir / "principle_reference_nearest_neighbors.json", nearest_table.to_dict(orient="records"))
    save_dataframe(top_nearest_table, results_dir / "principle_top_reference_nearest_neighbors.parquet")
    save_json(results_dir / "principle_top_reference_nearest_neighbors.json", top_nearest_table.to_dict(orient="records"))
    save_dataframe(family_table, results_dir / "principle_reference_family_centroids.parquet")
    save_json(results_dir / "principle_reference_family_centroids.json", family_table.to_dict(orient="records"))
    save_dataframe(top_family_table, results_dir / "principle_top_reference_family_centroids.parquet")
    save_json(results_dir / "principle_top_reference_family_centroids.json", top_family_table.to_dict(orient="records"))
    save_json(results_dir / "principle_reference_diagnostics_summary.json", summary)
    save_markdown(
        results_dir / "principle_reference_diagnostics_report.md",
        format_report(summary, nearest_table, top_nearest_table, family_table, top_family_table),
    )

    print(f"Saved principle reference diagnostics to {results_dir}")


if __name__ == "__main__":
    main()
