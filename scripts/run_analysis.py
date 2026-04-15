from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src.analysis.analyzer import (
    build_baseline_matrix,
    build_pairwise_similarity_table,
    build_prompt_feature_matrix,
    compute_paraphrase_stability,
    compute_similarity_matrix,
    out_of_fold_logistic_accuracy,
    out_of_fold_regression_predictions,
    random_top_k_mean,
    top_k_mean,
    write_analysis_summary,
)
from src.utils.io import ensure_dir, load_config, resolve_path, save_dataframe, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(description="Run activation-based transfer analysis.")
    parser.add_argument("--config", default="configs/config.yaml")
    return parser.parse_args()


def format_report(summary, feature_keys_count, prompt_count):
    lines = [
        "# Final Report",
        "",
        "## Overview",
        "",
        f"- Number of prompt variants analyzed: {prompt_count}",
        f"- Activation feature blocks (task x layer x position): {feature_keys_count}",
        f"- Activation ridge R^2: {summary['prediction']['activation_ridge_r2']:.4f}",
        f"- Activation logistic accuracy: {summary['prediction']['activation_logistic_accuracy']:.4f}",
        "",
        "## Baselines",
        "",
        f"- Seen-accuracy ridge R^2: {summary['prediction']['seen_accuracy_ridge_r2']:.4f}",
        f"- Prompt-meta ridge R^2: {summary['prediction']['prompt_meta_ridge_r2']:.4f}",
        "",
        "## Selection",
        "",
        f"- Activation top-k unseen accuracy: {summary['selection']['activation_top_k_unseen_accuracy']:.4f}",
        f"- Seen-accuracy top-k unseen accuracy: {summary['selection']['seen_accuracy_top_k_unseen_accuracy']:.4f}",
        f"- Random top-k unseen accuracy: {summary['selection']['random_top_k_unseen_accuracy']:.4f}",
        "",
        "## Stability",
        "",
        f"- Paraphrase pairs scored: {summary['stability']['num_pairs']}",
        f"- Mean paraphrase activation cosine: {summary['stability']['mean_activation_cosine']:.4f}",
        "",
    ]
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

    analysis_table = prompt_meta.merge(
        eval_prompt_summary,
        on=[
            "prompt_id",
            "group_id",
            "variant",
            "source",
            "prompt_text",
            "prompt_length_chars",
            "prompt_length_words",
        ],
        how="inner",
    )
    valid_prompt_ids = analysis_table["prompt_id"].tolist()
    mask = prompt_meta["prompt_id"].isin(valid_prompt_ids).to_numpy()
    prompt_features = prompt_features[mask]
    prompt_meta = prompt_meta[mask].reset_index(drop=True)
    analysis_table = analysis_table.reset_index(drop=True)

    similarity_matrix = compute_similarity_matrix(prompt_features)
    similarity_df = build_pairwise_similarity_table(prompt_meta, similarity_matrix)
    save_dataframe(similarity_df, results_dir / "similarity_pairs.parquet")

    stability_df = compute_paraphrase_stability(
        prompt_meta=prompt_meta,
        prompt_features=prompt_features,
        eval_prompt_summary=analysis_table,
    )
    save_dataframe(stability_df, results_dir / "paraphrase_stability.parquet")

    y_reg = analysis_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    y_cls = (y_reg >= np.nanmedian(y_reg)).astype(np.int64)

    activation_pred, activation_r2 = out_of_fold_regression_predictions(
        prompt_features,
        y_reg,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=config.get("seed", 42),
    )
    _, activation_logistic_acc = out_of_fold_logistic_accuracy(
        prompt_features,
        y_cls,
        c_value=config["analysis"]["logistic_c"],
        n_splits=config["analysis"]["n_splits"],
        random_state=config.get("seed", 42),
    )

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
    )
    _, prompt_meta_r2 = out_of_fold_regression_predictions(
        baseline_features["prompt_meta"],
        y_reg,
        alpha=config["analysis"]["ridge_alpha"],
        n_splits=config["analysis"]["n_splits"],
        random_state=config.get("seed", 42),
    )

    if "apo_rank" in baseline_features:
        _, apo_r2 = out_of_fold_regression_predictions(
            baseline_features["apo_rank"],
            y_reg,
            alpha=config["analysis"]["ridge_alpha"],
            n_splits=config["analysis"]["n_splits"],
            random_state=config.get("seed", 42),
        )
    else:
        apo_r2 = float("nan")

    top_k = config["analysis"]["top_k"]
    selection_summary = {
        "activation_top_k_unseen_accuracy": top_k_mean(activation_pred, y_reg, top_k),
        "seen_accuracy_top_k_unseen_accuracy": top_k_mean(
            analysis_table["seen_mean_accuracy"].to_numpy(dtype=np.float64),
            y_reg,
            top_k,
        ),
        "random_top_k_unseen_accuracy": random_top_k_mean(
            y_reg,
            top_k,
            trials=config["analysis"]["random_trials"],
            random_state=config.get("seed", 42),
        ),
    }

    summary = {
        "prediction": {
            "activation_ridge_r2": float(activation_r2),
            "activation_logistic_accuracy": float(activation_logistic_acc),
            "seen_accuracy_ridge_r2": float(seen_r2),
            "prompt_meta_ridge_r2": float(prompt_meta_r2),
            "apo_rank_ridge_r2": float(apo_r2),
        },
        "selection": selection_summary,
        "stability": {
            "num_pairs": int(len(stability_df)),
            "mean_activation_cosine": float(stability_df["activation_cosine"].mean()) if not stability_df.empty else float("nan"),
        },
    }

    write_analysis_summary(results_dir / "analysis_summary.json", summary)
    save_markdown(
        results_dir / "final_report.md",
        format_report(summary, feature_keys_count=len(feature_keys), prompt_count=len(prompt_meta)),
    )
    save_dataframe(analysis_table, results_dir / "analysis_prompt_table.parquet")

    print(f"Saved analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()

