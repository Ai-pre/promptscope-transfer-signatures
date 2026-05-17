from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import (
    build_prompt_feature_matrix,
    merge_prompt_features_with_eval,
    out_of_fold_regression_predictions,
)
from src.utils.io import ensure_dir, load_config, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Explain what activation-based performance predictions are selecting: "
            "top prompts, component-level scores, and length-adjusted diagnostic correlations."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="activation_diagnostics")
    parser.add_argument("--top-n", type=int, default=15)
    parser.add_argument("--slice-type", default="best", choices=["full", "best"])
    return parser.parse_args()


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def residualize(values, covariates):
    values = np.asarray(values, dtype=np.float64)
    covariates = np.asarray(covariates, dtype=np.float64)
    mask = np.isfinite(values) & np.all(np.isfinite(covariates), axis=1)
    residuals = np.full(len(values), np.nan, dtype=np.float64)
    if mask.sum() < 3:
        return residuals
    model = LinearRegression()
    model.fit(covariates[mask], values[mask])
    residuals[mask] = values[mask] - model.predict(covariates[mask])
    return residuals


def load_prompt_metadata(config):
    project_root = config["_project_root"]
    prompts_path = resolve_path(project_root, config["paths"]["prompts"])
    rows = []
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                rows.append(row)
    if not rows:
        return pd.DataFrame()
    meta = pd.DataFrame(rows)
    keep = [
        "id",
        "principle_components_json",
        "length_control_role",
        "principle_family",
        "paper_title",
        "source_note",
    ]
    keep = [column for column in keep if column in meta.columns]
    meta = meta[keep].rename(columns={"id": "prompt_id"})
    return meta


def attach_generation_diagnostics(eval_prompt_summary, eval_dir):
    if {"mean_prediction_words", "final_answer_marker_rate"}.issubset(eval_prompt_summary.columns):
        return eval_prompt_summary

    sample_path = eval_dir / "eval_results.parquet"
    if not sample_path.exists():
        output = eval_prompt_summary.copy()
        if "mean_prediction_words" not in output.columns:
            output["mean_prediction_words"] = np.nan
        if "final_answer_marker_rate" not in output.columns:
            output["final_answer_marker_rate"] = np.nan
        return output

    sample_df = pd.read_parquet(sample_path)
    prediction_text = sample_df["prediction"].fillna("").astype(str)
    sample_df = sample_df.copy()
    sample_df["prediction_words"] = prediction_text.str.split().str.len()
    sample_df["has_final_answer_marker"] = prediction_text.str.contains(
        "FINAL ANSWER",
        case=False,
        regex=False,
    )

    diagnostics = (
        sample_df.groupby(["prompt_id", "group_id", "variant", "source"], dropna=False)
        .agg(
            mean_prediction_words=("prediction_words", "mean"),
            final_answer_marker_rate=("has_final_answer_marker", "mean"),
        )
        .reset_index()
    )
    return eval_prompt_summary.merge(
        diagnostics,
        on=["prompt_id", "group_id", "variant", "source"],
        how="left",
    )


def parse_components(value):
    if isinstance(value, list):
        return value
    if not isinstance(value, str) or not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def coalesce_columns(frame, output_column, candidates):
    existing = [column for column in candidates if column in frame.columns]
    if not existing:
        return frame
    series = frame[existing[0]]
    for column in existing[1:]:
        series = series.combine_first(frame[column])
    frame[output_column] = series
    return frame


def select_best_slice(results_dir):
    slice_path = results_dir / "slice_analysis.parquet"
    if not slice_path.exists():
        return None
    table = pd.read_parquet(slice_path)
    if table.empty:
        return None
    candidates = table[table["slice_type"] == "layer_position"].copy()
    if candidates.empty:
        candidates = table.copy()
    ranked = candidates.sort_values(
        ["activation_top_k_unseen_accuracy", "activation_ridge_r2"],
        ascending=[False, False],
    )
    return ranked.iloc[0].to_dict()


def build_features(config, activation_summary, summary_vectors, results_dir, slice_type):
    sliced = activation_summary.copy()
    selected_slice = None
    if slice_type == "best":
        selected_slice = select_best_slice(results_dir)
        if selected_slice is not None:
            layer = selected_slice.get("layer")
            position = selected_slice.get("position")
            if layer is not None and not pd.isna(layer):
                sliced = sliced[sliced["layer"] == layer]
            if position is not None and not pd.isna(position):
                sliced = sliced[sliced["position"] == position]

    features, prompt_meta, feature_keys = build_prompt_feature_matrix(
        activation_summary_df=sliced,
        summary_vectors=summary_vectors,
        tasks=config["tasks"]["seen"],
    )
    return features, prompt_meta, feature_keys, selected_slice


def component_effects(score_table):
    all_components = sorted(
        {
            component
            for components in score_table["principle_components"]
            for component in components
        }
    )
    rows = []
    for component in all_components:
        present = score_table[
            score_table["principle_components"].apply(lambda components: component in components)
        ]
        absent = score_table[
            score_table["principle_components"].apply(lambda components: component not in components)
        ]
        if present.empty or absent.empty:
            continue
        rows.append(
            {
                "component": component,
                "present_count": int(len(present)),
                "absent_count": int(len(absent)),
                "delta_actual_unseen_accuracy": float(
                    present["unseen_mean_accuracy"].mean()
                    - absent["unseen_mean_accuracy"].mean()
                ),
                "delta_activation_predicted_accuracy": float(
                    present["activation_pred_unseen"].mean()
                    - absent["activation_pred_unseen"].mean()
                ),
                "delta_activation_error": float(
                    present["activation_prediction_error"].mean()
                    - absent["activation_prediction_error"].mean()
                ),
                "delta_mean_prediction_words": float(
                    present["mean_prediction_words"].mean()
                    - absent["mean_prediction_words"].mean()
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "component",
                "present_count",
                "absent_count",
                "delta_actual_unseen_accuracy",
                "delta_activation_predicted_accuracy",
                "delta_activation_error",
                "delta_mean_prediction_words",
            ]
        )
    return pd.DataFrame(rows).sort_values(
        ["delta_activation_predicted_accuracy", "delta_actual_unseen_accuracy"],
        ascending=[False, False],
    )


def role_effects(score_table):
    if "length_control_role" not in score_table.columns:
        return pd.DataFrame()
    grouped = (
        score_table.groupby("length_control_role", dropna=False)
        .agg(
            prompt_count=("prompt_id", "count"),
            actual_unseen_accuracy=("unseen_mean_accuracy", "mean"),
            activation_predicted_accuracy=("activation_pred_unseen", "mean"),
            activation_prediction_error=("activation_prediction_error", "mean"),
            mean_prediction_words=("mean_prediction_words", "mean"),
        )
        .reset_index()
        .sort_values(
            ["activation_predicted_accuracy", "actual_unseen_accuracy"],
            ascending=[False, False],
        )
    )
    return grouped


def format_prompt_rows(rows):
    lines = []
    for row in rows.itertuples(index=False):
        lines.append(
            "- "
            f"{row.prompt_id}: act_pred={row.activation_pred_unseen:.4f}, "
            f"actual={row.unseen_mean_accuracy:.4f}, "
            f"seen={row.seen_mean_accuracy:.4f}, "
            f"words={row.mean_prediction_words:.1f}"
        )
    return lines


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    outputs_root = resolve_path(project_root, config["paths"]["outputs_dir"])
    eval_dir = outputs_root / "eval"
    act_dir = outputs_root / "activations"
    results_dir = outputs_root / "results"
    out_dir = ensure_dir(outputs_root / args.output_subdir)

    eval_prompt_summary = pd.read_parquet(eval_dir / "eval_prompt_summary.parquet")
    eval_prompt_summary = attach_generation_diagnostics(eval_prompt_summary, eval_dir)
    activation_summary = pd.read_parquet(act_dir / "activation_summary.parquet")
    summary_vectors = np.load(act_dir / "activation_vectors.npz")["summary_vectors"]

    features, prompt_meta, feature_keys, selected_slice = build_features(
        config,
        activation_summary,
        summary_vectors,
        results_dir,
        args.slice_type,
    )
    features, prompt_meta, analysis_table = merge_prompt_features_with_eval(
        prompt_meta,
        features,
        eval_prompt_summary,
    )

    y = analysis_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    group_ids = analysis_table["group_id"].to_numpy()
    activation_pred, activation_r2 = out_of_fold_regression_predictions(
        features,
        y,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=config.get("seed", 42),
        groups=group_ids,
    )

    score_table = analysis_table.copy()
    score_table["activation_pred_unseen"] = activation_pred
    score_table["activation_prediction_error"] = (
        score_table["unseen_mean_accuracy"] - score_table["activation_pred_unseen"]
    )
    score_table["activation_rank"] = score_table["activation_pred_unseen"].rank(
        ascending=False,
        method="min",
    )
    score_table["actual_unseen_rank"] = score_table["unseen_mean_accuracy"].rank(
        ascending=False,
        method="min",
    )

    prompt_meta_extra = load_prompt_metadata(config)
    if not prompt_meta_extra.empty:
        score_table = score_table.merge(prompt_meta_extra, on="prompt_id", how="left")
    score_table = coalesce_columns(
        score_table,
        "principle_components_json",
        [
            "principle_components_json",
            "principle_components_json_x",
            "principle_components_json_y",
            "principle_components_json_meta",
        ],
    )
    score_table = coalesce_columns(
        score_table,
        "length_control_role",
        [
            "length_control_role",
            "length_control_role_x",
            "length_control_role_y",
            "taxonomy_role",
            "taxonomy_role_x",
            "taxonomy_role_y",
        ],
    )
    if "principle_components_json" not in score_table.columns:
        score_table["principle_components_json"] = "[]"
    score_table["principle_components"] = score_table["principle_components_json"].apply(
        parse_components
    )

    covariate_columns = [
        column
        for column in [
            "mean_prediction_words",
            "prompt_length_words",
            "final_answer_marker_rate",
        ]
        if column in score_table.columns
    ]
    length_residual_corr = float("nan")
    if covariate_columns:
        covariates = score_table[covariate_columns].to_numpy(dtype=np.float64)
        score_table["activation_pred_length_residual"] = residualize(
            score_table["activation_pred_unseen"].to_numpy(dtype=np.float64),
            covariates,
        )
        score_table["actual_unseen_length_residual"] = residualize(
            score_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64),
            covariates,
        )
        length_residual_corr = pearson(
            score_table["activation_pred_length_residual"],
            score_table["actual_unseen_length_residual"],
        )

    comp_table = component_effects(score_table)
    role_table = role_effects(score_table)
    top_activation = score_table.sort_values("activation_pred_unseen", ascending=False).head(
        args.top_n
    )
    top_actual = score_table.sort_values("unseen_mean_accuracy", ascending=False).head(args.top_n)
    false_positive = score_table.sort_values(
        "activation_prediction_error",
        ascending=True,
    ).head(args.top_n)
    false_negative = score_table.sort_values(
        "activation_prediction_error",
        ascending=False,
    ).head(args.top_n)

    summary = {
        "num_prompts": int(len(score_table)),
        "slice_type": args.slice_type,
        "selected_slice": selected_slice,
        "feature_blocks": int(len(feature_keys)),
        "activation_oof_r2": float(activation_r2),
        "activation_pred_pearson_actual": pearson(
            score_table["activation_pred_unseen"],
            score_table["unseen_mean_accuracy"],
        ),
        "activation_pred_pearson_seen": pearson(
            score_table["activation_pred_unseen"],
            score_table["seen_mean_accuracy"],
        ),
        "activation_pred_pearson_prediction_length": pearson(
            score_table["activation_pred_unseen"],
            score_table["mean_prediction_words"],
        )
        if "mean_prediction_words" in score_table.columns
        else float("nan"),
        "length_adjusted_activation_actual_residual_pearson": length_residual_corr,
        "top_activation_prompt_ids": top_activation["prompt_id"].tolist(),
        "top_actual_prompt_ids": top_actual["prompt_id"].tolist(),
    }

    save_dataframe(score_table, out_dir / "activation_prompt_scores.parquet")
    save_dataframe(comp_table, out_dir / "activation_component_effects.parquet")
    save_dataframe(role_table, out_dir / "activation_role_effects.parquet")
    save_json(out_dir / "activation_diagnostics_summary.json", summary)

    report = [
        "# Activation Diagnostics",
        "",
        "## Summary",
        "",
        f"- Prompt count: {summary['num_prompts']}",
        f"- Slice mode: {summary['slice_type']}",
        f"- Feature blocks: {summary['feature_blocks']}",
        f"- Activation OOF ridge R^2: {summary['activation_oof_r2']:.4f}",
        f"- Activation-predicted vs actual Pearson: {summary['activation_pred_pearson_actual']:.4f}",
        f"- Activation-predicted vs output length Pearson: {summary['activation_pred_pearson_prediction_length']:.4f}",
        f"- Length-adjusted activation/actual residual Pearson: {summary['length_adjusted_activation_actual_residual_pearson']:.4f}",
        "",
        "## Top Activation-Selected Prompts",
        "",
        *format_prompt_rows(top_activation),
        "",
        "## Top Actual Prompts",
        "",
        *format_prompt_rows(top_actual),
        "",
        "## Components Ranked By Activation-Predicted Accuracy",
        "",
        *[
            "- "
            f"{row.component}: delta_act_pred={row.delta_activation_predicted_accuracy:.4f}, "
            f"delta_actual={row.delta_actual_unseen_accuracy:.4f}, "
            f"delta_words={row.delta_mean_prediction_words:.1f}, "
            f"n={row.present_count}"
            for row in comp_table.head(args.top_n).itertuples(index=False)
        ],
        "",
        "## Most Overpredicted By Activation",
        "",
        *format_prompt_rows(false_positive),
        "",
        "## Most Underpredicted By Activation",
        "",
        *format_prompt_rows(false_negative),
        "",
    ]
    save_markdown(out_dir / "activation_diagnostics_report.md", "\n".join(report))
    print(f"Saved activation diagnostics to {out_dir}")


if __name__ == "__main__":
    main()
