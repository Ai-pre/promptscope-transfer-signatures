from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import (
    build_prompt_feature_matrix,
    cosine,
    merge_prompt_features_with_eval,
)
from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze principle-probe prompts against reference prompt results.")
    parser.add_argument("--config", default="configs/config.gpu_principle.yaml")
    parser.add_argument("--reference-config", default="configs/config.gpu_general_mixed.yaml")
    parser.add_argument("--top-reference-prompts", type=int, default=5)
    return parser.parse_args()


def parse_component_list(value):
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


def load_run_artifacts(config):
    project_root = config["_project_root"]
    outputs_root = resolve_path(project_root, config["paths"]["outputs_dir"])
    eval_dir = outputs_root / "eval"
    act_dir = outputs_root / "activations"
    results_dir = outputs_root / "results"
    return {
        "outputs_root": outputs_root,
        "eval_prompt_summary": pd.read_parquet(eval_dir / "eval_prompt_summary.parquet"),
        "activation_summary": pd.read_parquet(act_dir / "activation_summary.parquet"),
        "summary_vectors": np.load(act_dir / "activation_vectors.npz")["summary_vectors"],
        "slice_analysis": pd.read_parquet(results_dir / "slice_analysis.parquet") if (results_dir / "slice_analysis.parquet").exists() else pd.DataFrame(),
    }


def select_reference_slice(slice_analysis: pd.DataFrame):
    if slice_analysis.empty:
        return None
    layer_position = slice_analysis[slice_analysis["slice_type"] == "layer_position"].copy()
    if layer_position.empty:
        layer_position = slice_analysis.copy()
    ranked = layer_position.sort_values(
        ["activation_ridge_r2", "activation_top_k_unseen_accuracy"],
        ascending=[False, False],
    )
    return ranked.iloc[0].to_dict()


def build_slice_feature_table(activation_summary, summary_vectors, eval_prompt_summary, seen_tasks, *, layer, position):
    sliced = activation_summary.copy()
    sliced = sliced[sliced["task"].isin(seen_tasks)]
    if layer is not None and not (isinstance(layer, float) and np.isnan(layer)):
        sliced = sliced[sliced["layer"] == layer]
    if position is not None and not (isinstance(position, float) and pd.isna(position)):
        sliced = sliced[sliced["position"] == position]
    features, prompt_meta, feature_keys = build_prompt_feature_matrix(
        activation_summary_df=sliced,
        summary_vectors=summary_vectors,
        tasks=seen_tasks,
    )
    features, prompt_meta, analysis_table = merge_prompt_features_with_eval(
        prompt_meta,
        features,
        eval_prompt_summary,
    )
    return features, prompt_meta, analysis_table, feature_keys


def build_reference_centroid(reference_table, reference_features, top_n: int):
    filtered = reference_table[reference_table["source"] != "base"].copy()
    ranked = filtered.sort_values(
        ["unseen_mean_accuracy", "overall_accuracy", "seen_mean_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top_table = ranked.head(min(top_n, len(ranked)))
    if top_table.empty:
        return None, top_table
    index_by_prompt = {prompt_id: idx for idx, prompt_id in enumerate(reference_table["prompt_id"].tolist())}
    vectors = [reference_features[index_by_prompt[prompt_id]] for prompt_id in top_table["prompt_id"]]
    centroid = np.mean(np.stack(vectors), axis=0)
    return centroid, top_table


def add_reference_similarity(prompt_table, prompt_features, reference_centroid):
    if reference_centroid is None:
        prompt_table = prompt_table.copy()
        prompt_table["reference_centroid_cosine"] = np.nan
        return prompt_table

    similarities = [cosine(vector, reference_centroid) for vector in prompt_features]
    prompt_table = prompt_table.copy()
    prompt_table["reference_centroid_cosine"] = similarities
    return prompt_table


def build_component_columns(prompt_table):
    prompt_table = prompt_table.copy()
    prompt_table["principle_components"] = prompt_table["principle_components_json"].apply(parse_component_list)
    all_components = sorted({component for items in prompt_table["principle_components"] for component in items})
    for component in all_components:
        prompt_table[f"has_{component}"] = prompt_table["principle_components"].apply(lambda items: int(component in items))
    prompt_table["is_lightweight"] = (prompt_table["complexity_level"] != "heavy").astype(int)
    prompt_table["is_heavy"] = (prompt_table["complexity_level"] == "heavy").astype(int)
    return prompt_table, all_components


def build_component_effects(prompt_table, all_components):
    rows = []
    targets = ["seen_mean_accuracy", "unseen_mean_accuracy", "overall_accuracy", "transfer_gap", "reference_centroid_cosine"]
    for component in all_components + ["lightweight", "heavy"]:
        column = f"has_{component}" if component in all_components else f"is_{component}"
        present = prompt_table[prompt_table[column] == 1]
        absent = prompt_table[prompt_table[column] == 0]
        if present.empty or absent.empty:
            continue
        row = {
            "component": component,
            "present_count": int(len(present)),
            "absent_count": int(len(absent)),
        }
        for target in targets:
            row[f"present_{target}"] = float(present[target].mean())
            row[f"absent_{target}"] = float(absent[target].mean())
            row[f"delta_{target}"] = row[f"present_{target}"] - row[f"absent_{target}"]
        rows.append(row)
    return pd.DataFrame(rows)


def fit_component_linear_model(prompt_table, all_components, target_column):
    feature_columns = [f"has_{component}" for component in all_components] + ["is_heavy"]
    X = prompt_table[feature_columns].to_numpy(dtype=np.float64)
    X = np.hstack([np.ones((len(X), 1), dtype=np.float64), X])
    y = prompt_table[target_column].to_numpy(dtype=np.float64)
    coefficients, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
    names = ["intercept"] + feature_columns
    return {name: float(value) for name, value in zip(names, coefficients)}


def build_named_contrasts(prompt_table):
    by_id = {row["prompt_id"]: row for _, row in prompt_table.iterrows()}

    contrasts = [
        ("expert_vs_plain", "principle_expert_only", "principle_plain"),
        ("reasoning_vs_plain", "principle_reasoning_only", "principle_plain"),
        ("format_vs_plain", "principle_format_only", "principle_plain"),
        ("expert_reasoning_format_vs_plain", "principle_expert_reasoning_format", "principle_plain"),
        ("expert_reasoning_format_vs_expert_reasoning", "principle_expert_reasoning_format", "principle_expert_reasoning"),
        ("expert_reasoning_format_verify_vs_expert_reasoning_format", "principle_expert_reasoning_format_verify", "principle_expert_reasoning_format"),
        ("verbose_control_vs_lightweight_main", "principle_verbose_expert_reasoning_format", "principle_expert_reasoning_format"),
        ("multiagent_control_vs_lightweight_main", "principle_multiagent_scaffold", "principle_expert_reasoning_format"),
        ("careful_vs_plain", "principle2_careful", "principle2_plain"),
        ("format_vs_plain_refined", "principle2_format_only", "principle2_plain"),
        ("careful_format_vs_plain", "principle2_careful_format", "principle2_plain"),
        ("check_vs_plain", "principle2_check_only", "principle2_plain"),
        ("careful_check_vs_careful", "principle2_careful_check", "principle2_careful"),
        ("careful_format_check_vs_careful_format", "principle2_careful_format_check", "principle2_careful_format"),
        ("concise_format_vs_format", "principle2_concise_format", "principle2_format_only"),
        ("soft_reason_vs_plain", "principle2_soft_reason", "principle2_plain"),
        ("soft_reason_format_vs_format", "principle2_soft_reason_format", "principle2_format_only"),
        ("hard_reason_vs_plain_refined", "principle2_hard_reason", "principle2_plain"),
        ("strong_expert_vs_plain_refined", "principle2_strong_expert", "principle2_plain"),
        ("verbose_control_vs_careful_format", "principle2_verbose_control", "principle2_careful_format"),
        ("multiagent_control_vs_careful_format", "principle2_multiagent_control", "principle2_careful_format"),
        ("boundary_concise_vs_plain", "principle3_concise", "principle3_plain"),
        ("boundary_careful_vs_plain", "principle3_careful", "principle3_plain"),
        ("boundary_check_vs_plain", "principle3_check", "principle3_plain"),
        ("boundary_concise_format_vs_plain", "principle3_concise_format", "principle3_plain"),
        ("boundary_careful_format_vs_plain", "principle3_careful_format", "principle3_plain"),
        ("boundary_careful_check_vs_careful", "principle3_careful_check", "principle3_careful"),
        ("boundary_careful_format_check_vs_careful_format", "principle3_careful_format_check", "principle3_careful_format"),
        ("boundary_soft_reason_vs_plain", "principle3_soft_reason", "principle3_plain"),
        ("boundary_soft_reason_format_vs_concise_format", "principle3_soft_reason_format", "principle3_concise_format"),
        ("boundary_concise_careful_format_vs_concise_format", "principle3_concise_careful_format", "principle3_concise_format"),
        ("boundary_concise_careful_check_vs_careful_check", "principle3_concise_careful_check", "principle3_careful_check"),
    ]

    rows = []
    for contrast_name, left_id, right_id in contrasts:
        if left_id not in by_id or right_id not in by_id:
            continue
        left = by_id[left_id]
        right = by_id[right_id]
        rows.append(
            {
                "contrast": contrast_name,
                "left_prompt_id": left_id,
                "right_prompt_id": right_id,
                "delta_seen_mean_accuracy": float(left["seen_mean_accuracy"] - right["seen_mean_accuracy"]),
                "delta_unseen_mean_accuracy": float(left["unseen_mean_accuracy"] - right["unseen_mean_accuracy"]),
                "delta_overall_accuracy": float(left["overall_accuracy"] - right["overall_accuracy"]),
                "delta_transfer_gap": float(left["transfer_gap"] - right["transfer_gap"]),
                "delta_reference_centroid_cosine": float(left["reference_centroid_cosine"] - right["reference_centroid_cosine"]),
            }
        )
    return rows


def format_report(principle_table, component_effects, contrasts, reference_slice, reference_top):
    best_unseen = principle_table.sort_values(["unseen_mean_accuracy", "reference_centroid_cosine"], ascending=[False, False]).iloc[0]
    best_similarity = principle_table.sort_values(["reference_centroid_cosine", "unseen_mean_accuracy"], ascending=[False, False]).iloc[0]

    lines = [
        "# Principle Probe Report",
        "",
        "## Reference Slice",
        "",
    ]
    if reference_slice is None:
        lines.extend(["- No reference slice available.", ""])
    else:
        lines.extend(
            [
                f"- Slice type: {reference_slice['slice_type']}",
                f"- Layer: {reference_slice['layer']}",
                f"- Position: {reference_slice['position']}",
                f"- Reference activation ridge R^2: {reference_slice['activation_ridge_r2']:.4f}",
                "",
            ]
        )
    if not reference_top.empty:
        lines.extend(
            [
                "## Reference Top Prompts",
                "",
                *[
                    f"- {row.prompt_id}: unseen={row.unseen_mean_accuracy:.4f}, seen={row.seen_mean_accuracy:.4f}"
                    for row in reference_top.itertuples(index=False)
                ],
                "",
            ]
        )

    lines.extend(
        [
            "## Best Principle Probes",
            "",
            f"- Best unseen accuracy: {best_unseen['prompt_id']} (unseen={best_unseen['unseen_mean_accuracy']:.4f}, cosine={best_unseen['reference_centroid_cosine']:.4f})",
            f"- Closest to reference centroid: {best_similarity['prompt_id']} (cosine={best_similarity['reference_centroid_cosine']:.4f}, unseen={best_similarity['unseen_mean_accuracy']:.4f})",
            "",
        ]
    )

    if not component_effects.empty:
        top_components = component_effects.sort_values("delta_unseen_mean_accuracy", ascending=False).head(5)
        lines.extend(
            [
                "## Top Positive Components",
                "",
                *[
                    f"- {row.component}: delta unseen={row.delta_unseen_mean_accuracy:.4f}, delta cosine={row.delta_reference_centroid_cosine:.4f}"
                    for row in top_components.itertuples(index=False)
                ],
                "",
            ]
        )

    if contrasts:
        lines.extend(
            [
                "## Key Contrasts",
                "",
                *[
                    f"- {row['contrast']}: delta unseen={row['delta_unseen_mean_accuracy']:.4f}, delta cosine={row['delta_reference_centroid_cosine']:.4f}"
                    for row in contrasts
                ],
                "",
            ]
        )

    return "\n".join(lines)


def main():
    args = parse_args()
    principle_config = load_config(args.config)
    reference_config = load_config(args.reference_config)

    principle_prompts = load_prompts(resolve_path(principle_config["_project_root"], principle_config["paths"]["prompts"]))
    principle_prompt_meta = pd.DataFrame(principle_prompts).rename(columns={"id": "prompt_id"})

    principle_artifacts = load_run_artifacts(principle_config)
    reference_artifacts = load_run_artifacts(reference_config)

    reference_slice = select_reference_slice(reference_artifacts["slice_analysis"])
    if reference_slice is None:
        raise ValueError("Reference slice analysis is empty; run reference analysis before principle analysis.")

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
    principle_prompt_ids = set(principle_prompt_meta["prompt_id"].tolist())
    principle_mask = principle_table["prompt_id"].isin(principle_prompt_ids)
    principle_features = principle_features[principle_mask.to_numpy()]
    principle_table = principle_table[principle_mask].reset_index(drop=True)
    if principle_table.empty:
        raise ValueError(
            "No principle prompts were found in the selected run outputs. "
            "Check that the config prompts file matches the eval/activation outputs."
        )
    reference_features, _, reference_table, _ = build_slice_feature_table(
        reference_artifacts["activation_summary"],
        reference_artifacts["summary_vectors"],
        reference_artifacts["eval_prompt_summary"],
        reference_config["tasks"]["seen"],
        layer=slice_layer,
        position=slice_position,
    )

    principle_table = principle_table.merge(
        principle_prompt_meta[
            [
                "prompt_id",
                "principle_family",
                "principle_components_json",
                "complexity_level",
                "hypothesis_role",
                "contrast_group",
                "source_note",
            ]
        ],
        on="prompt_id",
        how="left",
        suffixes=("", "_meta"),
    )

    reference_centroid, reference_top = build_reference_centroid(
        reference_table=reference_table,
        reference_features=reference_features,
        top_n=args.top_reference_prompts,
    )
    principle_table = add_reference_similarity(principle_table, principle_features, reference_centroid)
    principle_table, all_components = build_component_columns(principle_table)

    component_effects = build_component_effects(principle_table, all_components)
    linear_weights = {
        "unseen_mean_accuracy": fit_component_linear_model(principle_table, all_components, "unseen_mean_accuracy"),
        "reference_centroid_cosine": fit_component_linear_model(principle_table, all_components, "reference_centroid_cosine"),
    }
    contrasts = build_named_contrasts(principle_table)

    outputs_root = resolve_path(principle_config["_project_root"], principle_config["paths"]["outputs_dir"])
    results_dir = ensure_dir(outputs_root / "principle_results")

    save_dataframe(principle_table, results_dir / "principle_prompt_table.parquet")
    save_json(results_dir / "principle_prompt_table.json", principle_table.to_dict(orient="records"))
    save_dataframe(component_effects, results_dir / "principle_component_effects.parquet")
    save_json(results_dir / "principle_component_effects.json", component_effects.to_dict(orient="records"))
    save_json(results_dir / "principle_linear_weights.json", linear_weights)
    save_json(results_dir / "principle_contrasts.json", contrasts)

    summary = {
        "reference_slice": reference_slice,
        "reference_top_prompts": reference_top["prompt_id"].tolist(),
        "best_unseen_probe": principle_table.sort_values(
            ["unseen_mean_accuracy", "reference_centroid_cosine"],
            ascending=[False, False],
        ).iloc[0]["prompt_id"],
        "best_similarity_probe": principle_table.sort_values(
            ["reference_centroid_cosine", "unseen_mean_accuracy"],
            ascending=[False, False],
        ).iloc[0]["prompt_id"],
        "component_count": len(all_components),
    }
    save_json(results_dir / "principle_summary.json", summary)
    save_markdown(
        results_dir / "principle_report.md",
        format_report(principle_table, component_effects, contrasts, reference_slice, reference_top),
    )

    print(f"Saved principle analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()
