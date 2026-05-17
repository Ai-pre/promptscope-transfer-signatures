from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import (
    build_prompt_feature_matrix,
    merge_prompt_features_with_eval,
    out_of_fold_regression_predictions,
    random_top_k_mean,
    top_k_mean,
)
from src.utils.io import ensure_dir, load_config, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test whether activation-based prompt selection remains useful after "
            "matching prompts by generated output length."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="length_matched_activation_selection")
    parser.add_argument("--slice-type", choices=["best", "full"], default="best")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--quantile-bins", type=int, default=3)
    parser.add_argument("--fixed-thresholds", default="25,45")
    parser.add_argument("--random-trials", type=int, default=None)
    parser.add_argument("--include-base", action="store_true")
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


def spearman(x, y):
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson(x_rank, y_rank)


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


def attach_generation_diagnostics(eval_prompt_summary, eval_dir):
    required = {"mean_prediction_words", "final_answer_marker_rate"}
    if required.issubset(eval_prompt_summary.columns):
        return eval_prompt_summary

    sample_path = eval_dir / "eval_results.parquet"
    output = eval_prompt_summary.copy()
    if not sample_path.exists():
        for column in required:
            if column not in output.columns:
                output[column] = np.nan
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
    return output.merge(
        diagnostics,
        on=["prompt_id", "group_id", "variant", "source"],
        how="left",
    )


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


def load_prompt_metadata(config):
    project_root = config["_project_root"]
    prompts_path = resolve_path(project_root, config["paths"]["prompts"])
    rows = []
    with prompts_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    meta = pd.DataFrame(rows)
    keep = [
        "id",
        "principle_components_json",
        "length_control_role",
        "taxonomy_role",
        "paper_title",
    ]
    keep = [column for column in keep if column in meta.columns]
    return meta[keep].rename(columns={"id": "prompt_id"})


def parse_fixed_thresholds(raw):
    thresholds = []
    for part in raw.split(","):
        part = part.strip()
        if part:
            thresholds.append(float(part))
    return sorted(thresholds)


def add_fixed_bins(table, thresholds):
    if len(thresholds) != 2:
        raise ValueError("--fixed-thresholds currently expects two comma-separated values.")
    low, high = thresholds
    bins = [-np.inf, low, high, np.inf]
    labels = [f"short<= {low:g}", f"mid({low:g},{high:g}]", f"long> {high:g}"]
    output = table.copy()
    output["fixed_length_bin"] = pd.cut(
        output["mean_prediction_words"],
        bins=bins,
        labels=labels,
        include_lowest=True,
    ).astype(str)
    return output


def add_quantile_bins(table, q):
    output = table.copy()
    ranked_lengths = output["mean_prediction_words"].rank(method="first")
    labels = [f"q{i + 1}" for i in range(q)]
    output["quantile_length_bin"] = pd.qcut(
        ranked_lengths,
        q=q,
        labels=labels,
        duplicates="drop",
    ).astype(str)
    return output


def top_prompt_ids(table, score_column, k):
    return (
        table.sort_values(score_column, ascending=False)
        .head(min(k, len(table)))["prompt_id"]
        .tolist()
    )


def evaluate_bins(table, bin_column, *, top_k, random_trials, random_state):
    rows = []
    for bin_name, group in table.groupby(bin_column, dropna=False):
        group = group.copy()
        if group.empty:
            continue
        k = min(top_k, len(group))
        activation_score = group["activation_pred_unseen"].to_numpy(dtype=np.float64)
        residual_score = group["activation_pred_residual"].to_numpy(dtype=np.float64)
        seen_score = group["seen_mean_accuracy"].to_numpy(dtype=np.float64)
        target = group["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
        residual_target = group["length_adjusted_accuracy_residual"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "binning": bin_column,
                "bin": str(bin_name),
                "prompt_count": int(len(group)),
                "top_k": int(k),
                "mean_prediction_words_min": float(group["mean_prediction_words"].min()),
                "mean_prediction_words_max": float(group["mean_prediction_words"].max()),
                "mean_unseen_accuracy": float(group["unseen_mean_accuracy"].mean()),
                "activation_pearson_actual": pearson(activation_score, target),
                "activation_spearman_actual": spearman(activation_score, target),
                "activation_top_k_unseen_accuracy": top_k_mean(activation_score, target, k),
                "seen_top_k_unseen_accuracy": top_k_mean(seen_score, target, k),
                "random_top_k_unseen_accuracy": random_top_k_mean(
                    target,
                    k,
                    trials=random_trials,
                    random_state=random_state,
                ),
                "activation_minus_random": top_k_mean(activation_score, target, k)
                - random_top_k_mean(
                    target,
                    k,
                    trials=random_trials,
                    random_state=random_state,
                ),
                "activation_minus_seen": top_k_mean(activation_score, target, k)
                - top_k_mean(seen_score, target, k),
                "residual_activation_pearson_actual_residual": pearson(
                    residual_score,
                    residual_target,
                ),
                "residual_activation_top_k_accuracy_residual": top_k_mean(
                    residual_score,
                    residual_target,
                    k,
                ),
                "random_top_k_accuracy_residual": random_top_k_mean(
                    residual_target,
                    k,
                    trials=random_trials,
                    random_state=random_state,
                ),
                "activation_top_prompt_ids": top_prompt_ids(group, "activation_pred_unseen", k),
                "seen_top_prompt_ids": top_prompt_ids(group, "seen_mean_accuracy", k),
                "actual_top_prompt_ids": top_prompt_ids(group, "unseen_mean_accuracy", k),
            }
        )
    return pd.DataFrame(rows)


def summarize_bins(bin_table):
    if bin_table.empty:
        return {}
    return {
        "num_bins": int(len(bin_table)),
        "mean_activation_minus_random": float(bin_table["activation_minus_random"].mean()),
        "mean_activation_minus_seen": float(bin_table["activation_minus_seen"].mean()),
        "weighted_activation_minus_random": float(
            np.average(bin_table["activation_minus_random"], weights=bin_table["prompt_count"])
        ),
        "weighted_activation_minus_seen": float(
            np.average(bin_table["activation_minus_seen"], weights=bin_table["prompt_count"])
        ),
        "mean_activation_pearson_actual": float(bin_table["activation_pearson_actual"].mean()),
        "mean_residual_activation_pearson_actual_residual": float(
            bin_table["residual_activation_pearson_actual_residual"].mean()
        ),
    }


def format_bin_rows(bin_table):
    lines = []
    for row in bin_table.itertuples(index=False):
        lines.append(
            "- "
            f"{row.binning}/{row.bin}: n={row.prompt_count}, "
            f"words=[{row.mean_prediction_words_min:.1f}, {row.mean_prediction_words_max:.1f}], "
            f"act_topk={row.activation_top_k_unseen_accuracy:.4f}, "
            f"seen_topk={row.seen_top_k_unseen_accuracy:.4f}, "
            f"random={row.random_top_k_unseen_accuracy:.4f}, "
            f"act-random={row.activation_minus_random:.4f}, "
            f"corr={row.activation_pearson_actual:.4f}"
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

    top_k = args.top_k or int(config["analysis"]["top_k"])
    random_trials = args.random_trials or int(config["analysis"]["random_trials"])
    random_state = int(config.get("seed", 42))

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

    if not args.include_base:
        mask = analysis_table["source"] != "base"
        analysis_table = analysis_table[mask].reset_index(drop=True)
        features = features[mask.to_numpy()]

    prompt_meta_extra = load_prompt_metadata(config)
    if not prompt_meta_extra.empty:
        analysis_table = analysis_table.merge(prompt_meta_extra, on="prompt_id", how="left")

    y = analysis_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    group_ids = analysis_table["group_id"].to_numpy()
    activation_pred, activation_r2 = out_of_fold_regression_predictions(
        features,
        y,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=random_state,
        groups=group_ids,
    )

    covariate_columns = [
        "mean_prediction_words",
        "prompt_length_words",
        "final_answer_marker_rate",
    ]
    covariate_columns = [column for column in covariate_columns if column in analysis_table.columns]
    covariates = analysis_table[covariate_columns].to_numpy(dtype=np.float64)
    y_residual = residualize(y, covariates)
    residual_pred, residual_r2 = out_of_fold_regression_predictions(
        features,
        y_residual,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=random_state,
        groups=group_ids,
    )

    score_table = analysis_table.copy()
    score_table["activation_pred_unseen"] = activation_pred
    score_table["length_adjusted_accuracy_residual"] = y_residual
    score_table["activation_pred_residual"] = residual_pred
    score_table = add_fixed_bins(score_table, parse_fixed_thresholds(args.fixed_thresholds))
    score_table = add_quantile_bins(score_table, args.quantile_bins)

    fixed_bins = evaluate_bins(
        score_table,
        "fixed_length_bin",
        top_k=top_k,
        random_trials=random_trials,
        random_state=random_state,
    )
    quantile_bins = evaluate_bins(
        score_table,
        "quantile_length_bin",
        top_k=top_k,
        random_trials=random_trials,
        random_state=random_state,
    )
    all_bins = pd.concat([fixed_bins, quantile_bins], ignore_index=True)

    summary = {
        "num_candidate_prompts": int(len(score_table)),
        "include_base": bool(args.include_base),
        "slice_type": args.slice_type,
        "selected_slice": selected_slice,
        "feature_blocks": int(len(feature_keys)),
        "top_k": int(top_k),
        "activation_oof_r2": float(activation_r2),
        "activation_pred_pearson_actual": pearson(activation_pred, y),
        "activation_pred_pearson_prediction_length": pearson(
            activation_pred,
            score_table["mean_prediction_words"],
        ),
        "length_adjusted_activation_oof_r2": float(residual_r2),
        "length_adjusted_activation_pred_pearson_residual": pearson(
            residual_pred,
            y_residual,
        ),
        "fixed_bin_summary": summarize_bins(fixed_bins),
        "quantile_bin_summary": summarize_bins(quantile_bins),
    }

    save_dataframe(score_table, out_dir / "length_matched_prompt_scores.parquet")
    save_dataframe(all_bins, out_dir / "length_matched_bin_results.parquet")
    save_json(out_dir / "length_matched_summary.json", summary)
    save_json(out_dir / "length_matched_bin_results.json", all_bins.to_dict(orient="records"))

    report = [
        "# Length-Matched Activation Selection",
        "",
        "## Summary",
        "",
        f"- Candidate prompts: {summary['num_candidate_prompts']}",
        f"- Include base prompt: {summary['include_base']}",
        f"- Slice mode: {summary['slice_type']}",
        f"- Feature blocks: {summary['feature_blocks']}",
        f"- Top-k per bin: {summary['top_k']}",
        f"- Activation OOF ridge R^2: {summary['activation_oof_r2']:.4f}",
        f"- Activation-predicted vs actual Pearson: {summary['activation_pred_pearson_actual']:.4f}",
        f"- Activation-predicted vs output length Pearson: {summary['activation_pred_pearson_prediction_length']:.4f}",
        f"- Length-adjusted activation OOF R^2: {summary['length_adjusted_activation_oof_r2']:.4f}",
        f"- Length-adjusted activation/residual Pearson: {summary['length_adjusted_activation_pred_pearson_residual']:.4f}",
        "",
        "## Fixed Length Bins",
        "",
        *format_bin_rows(fixed_bins),
        "",
        "## Quantile Length Bins",
        "",
        *format_bin_rows(quantile_bins),
        "",
    ]
    save_markdown(out_dir / "length_matched_report.md", "\n".join(report))
    print(f"Saved length-matched activation selection outputs to {out_dir}")


if __name__ == "__main__":
    main()
