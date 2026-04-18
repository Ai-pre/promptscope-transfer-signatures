from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import (
    augment_with_seen_score,
    build_slice_analysis_table,
    build_baseline_matrix,
    build_pairwise_similarity_table,
    build_prompt_feature_matrix,
    compute_paraphrase_stability,
    compute_similarity_matrix,
    evaluate_prediction_block,
    merge_prompt_features_with_eval,
    out_of_fold_regression_predictions,
    write_analysis_summary,
)
from src.utils.io import ensure_dir, load_config, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation-based transfer analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def format_report(summary, feature_keys_count, prompt_count, best_slice=None, best_hybrid_slice=None):
    lines = [
        "# Final Report",
        "",
        "## Overview",
        "",
        f"- Number of prompt variants analyzed: {prompt_count}",
        f"- Activation feature blocks (task x layer x position): {feature_keys_count}",
        f"- Activation ridge R^2: {summary['prediction']['activation_ridge_r2']:.4f}",
        f"- Activation logistic accuracy: {summary['prediction']['activation_logistic_accuracy']:.4f}",
        f"- Hybrid (seen + activation) ridge R^2: {summary['prediction']['hybrid_seen_activation_ridge_r2']:.4f}",
        f"- Hybrid (seen + activation) logistic accuracy: {summary['prediction']['hybrid_seen_activation_logistic_accuracy']:.4f}",
        "",
        "## Baselines",
        "",
        f"- Seen-accuracy ridge R^2: {summary['prediction']['seen_accuracy_ridge_r2']:.4f}",
        f"- Prompt-meta ridge R^2: {summary['prediction']['prompt_meta_ridge_r2']:.4f}",
        "",
        "## Selection",
        "",
        f"- Activation top-k unseen accuracy: {summary['selection']['activation_top_k_unseen_accuracy']:.4f}",
        f"- Hybrid (seen + activation) top-k unseen accuracy: {summary['selection']['hybrid_seen_activation_top_k_unseen_accuracy']:.4f}",
        f"- Seen-accuracy top-k unseen accuracy: {summary['selection']['seen_accuracy_top_k_unseen_accuracy']:.4f}",
        f"- Random top-k unseen accuracy: {summary['selection']['random_top_k_unseen_accuracy']:.4f}",
        "",
        "## Stability",
        "",
        f"- Paraphrase pairs scored: {summary['stability']['num_pairs']}",
        f"- Mean paraphrase activation cosine: {summary['stability']['mean_activation_cosine']:.4f}",
        "",
    ]
    if best_slice is not None:
        lines.extend(
            [
                "## Best Slice",
                "",
                f"- Slice type: {best_slice['slice_type']}",
                f"- Layer: {best_slice['layer']}",
                f"- Position: {best_slice['position']}",
                f"- Slice ridge R^2: {best_slice['activation_ridge_r2']:.4f}",
                f"- Slice top-k unseen accuracy: {best_slice['activation_top_k_unseen_accuracy']:.4f}",
                "",
            ]
        )
    if best_hybrid_slice is not None:
        lines.extend(
            [
                "## Best Hybrid Slice",
                "",
                f"- Slice type: {best_hybrid_slice['slice_type']}",
                f"- Layer: {best_hybrid_slice['layer']}",
                f"- Position: {best_hybrid_slice['position']}",
                f"- Hybrid slice ridge R^2: {best_hybrid_slice['hybrid_activation_ridge_r2']:.4f}",
                f"- Hybrid slice top-k unseen accuracy: {best_hybrid_slice['hybrid_activation_top_k_unseen_accuracy']:.4f}",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    outputs_root = resolve_path(project_root, config["paths"]["outputs_dir"])

    eval_dir = outputs_root / "eval"
    act_dir = outputs_root / "activations"
    results_dir = ensure_dir(outputs_root / "results")

    eval_prompt_summary = pd.read_parquet(eval_dir / "eval_prompt_summary.parquet")
    activation_summary = pd.read_parquet(act_dir / "activation_summary.parquet")
    activation_bundle = np.load(act_dir / "activation_vectors.npz")
    summary_vectors = activation_bundle["summary_vectors"]

    prompt_features, prompt_meta, feature_keys = build_prompt_feature_matrix(
        activation_summary_df=activation_summary,
        summary_vectors=summary_vectors,
        tasks=config["tasks"]["seen"],
    )

    prompt_features, prompt_meta, analysis_table = merge_prompt_features_with_eval(
        prompt_meta,
        prompt_features,
        eval_prompt_summary,
    )

    similarity_matrix = compute_similarity_matrix(prompt_features)
    similarity_df = build_pairwise_similarity_table(prompt_meta, similarity_matrix)
    save_dataframe(similarity_df, results_dir / "similarity_pairs.parquet")

    stability_df = compute_paraphrase_stability(
        prompt_meta=prompt_meta,
        prompt_features=prompt_features,
        eval_prompt_summary=analysis_table,
    )
    save_dataframe(stability_df, results_dir / "paraphrase_stability.parquet")

    prediction_metrics = evaluate_prediction_block(
        prompt_features,
        analysis_table,
        alpha=config["analysis"]["ridge_alpha"],
        c_value=config["analysis"]["logistic_c"],
        n_splits=config["analysis"]["n_splits"],
        top_k=config["analysis"]["top_k"],
        random_trials=config["analysis"]["random_trials"],
        random_state=config.get("seed", 42),
    )
    hybrid_features = augment_with_seen_score(prompt_features, analysis_table)
    hybrid_prediction_metrics = evaluate_prediction_block(
        hybrid_features,
        analysis_table,
        alpha=config["analysis"]["ridge_alpha"],
        c_value=config["analysis"]["logistic_c"],
        n_splits=config["analysis"]["n_splits"],
        top_k=config["analysis"]["top_k"],
        random_trials=config["analysis"]["random_trials"],
        random_state=config.get("seed", 42),
    )

    y_reg = analysis_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    group_ids = analysis_table["group_id"].to_numpy()

    baseline_features, _ = build_baseline_matrix(
        eval_prompt_summary=analysis_table,
        base_prompt_text=config["base_prompt"],
    )
    _, seen_r2 = out_of_fold_regression_predictions(
        baseline_features["seen_accuracy"],
        y_reg,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=config.get("seed", 42),
        groups=group_ids,
    )
    _, prompt_meta_r2 = out_of_fold_regression_predictions(
        baseline_features["prompt_meta"],
        y_reg,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=config.get("seed", 42),
        groups=group_ids,
    )

    if "apo_rank" in baseline_features:
        _, apo_r2 = out_of_fold_regression_predictions(
            baseline_features["apo_rank"],
            y_reg,
            alpha=config["analysis"]["ridge_alpha"],
            n_splits=config["analysis"]["n_splits"],
            random_state=config.get("seed", 42),
            groups=group_ids,
        )
    else:
        apo_r2 = float("nan")

    slice_analysis = build_slice_analysis_table(
        activation_summary_df=activation_summary,
        summary_vectors=summary_vectors,
        eval_prompt_summary=eval_prompt_summary,
        tasks=config["tasks"]["seen"],
        alpha=config["analysis"]["ridge_alpha"],
        c_value=config["analysis"]["logistic_c"],
        n_splits=config["analysis"]["n_splits"],
        top_k=config["analysis"]["top_k"],
        random_trials=config["analysis"]["random_trials"],
        random_state=config.get("seed", 42),
        include_seen_hybrid=True,
    )
    save_dataframe(slice_analysis, results_dir / "slice_analysis.parquet")
    save_json(
        results_dir / "slice_analysis.json",
        slice_analysis.to_dict(orient="records"),
    )

    best_slice = None
    if not slice_analysis.empty:
        ranked = slice_analysis.sort_values(
            ["activation_top_k_unseen_accuracy", "activation_ridge_r2"],
            ascending=[False, False],
        )
        best_slice = ranked.iloc[0].to_dict()
    best_hybrid_slice = None
    if not slice_analysis.empty and "hybrid_activation_top_k_unseen_accuracy" in slice_analysis.columns:
        hybrid_ranked = slice_analysis.sort_values(
            ["hybrid_activation_top_k_unseen_accuracy", "hybrid_activation_ridge_r2"],
            ascending=[False, False],
        )
        best_hybrid_slice = hybrid_ranked.iloc[0].to_dict()

    summary = {
        "prediction": {
            "activation_ridge_r2": prediction_metrics["activation_ridge_r2"],
            "activation_logistic_accuracy": prediction_metrics["activation_logistic_accuracy"],
            "hybrid_seen_activation_ridge_r2": hybrid_prediction_metrics["activation_ridge_r2"],
            "hybrid_seen_activation_logistic_accuracy": hybrid_prediction_metrics["activation_logistic_accuracy"],
            "seen_accuracy_ridge_r2": float(seen_r2),
            "prompt_meta_ridge_r2": float(prompt_meta_r2),
            "apo_rank_ridge_r2": float(apo_r2),
        },
        "selection": {
            "activation_top_k_unseen_accuracy": prediction_metrics["activation_top_k_unseen_accuracy"],
            "hybrid_seen_activation_top_k_unseen_accuracy": hybrid_prediction_metrics["activation_top_k_unseen_accuracy"],
            "seen_accuracy_top_k_unseen_accuracy": prediction_metrics["seen_accuracy_top_k_unseen_accuracy"],
            "random_top_k_unseen_accuracy": prediction_metrics["random_top_k_unseen_accuracy"],
        },
        "stability": {
            "num_pairs": int(len(stability_df)),
            "mean_activation_cosine": float(stability_df["activation_cosine"].mean()) if not stability_df.empty else float("nan"),
        },
        "best_slice": best_slice,
        "best_hybrid_slice": best_hybrid_slice,
    }

    write_analysis_summary(results_dir / "analysis_summary.json", summary)
    save_markdown(
        results_dir / "final_report.md",
        format_report(
            summary,
            feature_keys_count=len(feature_keys),
            prompt_count=len(prompt_meta),
            best_slice=best_slice,
            best_hybrid_slice=best_hybrid_slice,
        ),
    )
    save_dataframe(analysis_table, results_dir / "analysis_prompt_table.parquet")

    print(f"Saved analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()
