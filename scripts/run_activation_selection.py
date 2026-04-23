from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import (
    build_prompt_feature_matrix,
    cosine,
    merge_prompt_features_with_eval,
    random_top_k_mean,
    top_k_mean,
)
from src.utils.io import ensure_dir, load_config, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Validate activation-based prompt selection by ranking candidate prompts "
            "against a reference activation centroid."
        )
    )
    parser.add_argument("--reference-config", default="configs/config.gpu_general_mixed_clean.yaml")
    parser.add_argument("--candidate-config", default="configs/config.gpu_principle_boundary.yaml")
    parser.add_argument("--top-reference-prompts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output-subdir", default="activation_selection")
    return parser.parse_args()


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
        "slice_analysis": pd.read_parquet(results_dir / "slice_analysis.parquet"),
    }


def select_reference_slice(slice_analysis: pd.DataFrame):
    if slice_analysis.empty:
        raise ValueError("Reference slice_analysis is empty. Run reference run_analysis.py first.")

    candidates = slice_analysis[slice_analysis["slice_type"] == "layer_position"].copy()
    if candidates.empty:
        candidates = slice_analysis.copy()

    ranked = candidates.sort_values(
        ["activation_ridge_r2", "activation_top_k_unseen_accuracy"],
        ascending=[False, False],
    )
    return ranked.iloc[0].to_dict()


def build_slice_features(artifacts, config, *, layer, position):
    activation_summary = artifacts["activation_summary"]
    sliced = activation_summary[activation_summary["task"].isin(config["tasks"]["seen"])].copy()

    if layer is not None and not pd.isna(layer):
        sliced = sliced[sliced["layer"] == layer]
    if position is not None and not pd.isna(position):
        sliced = sliced[sliced["position"] == position]
    if sliced.empty:
        raise ValueError(f"No activation rows for layer={layer!r}, position={position!r}.")

    features, prompt_meta, feature_keys = build_prompt_feature_matrix(
        activation_summary_df=sliced,
        summary_vectors=artifacts["summary_vectors"],
        tasks=config["tasks"]["seen"],
    )
    features, prompt_meta, table = merge_prompt_features_with_eval(
        prompt_meta,
        features,
        artifacts["eval_prompt_summary"],
    )
    return features, table, feature_keys


def top_reference_centroid(reference_features, reference_table, top_n: int):
    filtered = reference_table[reference_table["source"] != "base"].copy()
    ranked = filtered.sort_values(
        ["unseen_mean_accuracy", "overall_accuracy", "seen_mean_accuracy"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    top_table = ranked.head(min(top_n, len(ranked)))
    if top_table.empty:
        raise ValueError("No non-base reference prompts are available for centroid construction.")

    index_by_prompt = {prompt_id: idx for idx, prompt_id in enumerate(reference_table["prompt_id"].tolist())}
    vectors = [reference_features[index_by_prompt[prompt_id]] for prompt_id in top_table["prompt_id"]]
    return np.mean(np.stack(vectors), axis=0), top_table


def rank_candidates(candidate_features, candidate_table, centroid):
    table = candidate_table[candidate_table["source"] != "base"].copy().reset_index(drop=True)
    keep_index = candidate_table[candidate_table["source"] != "base"].index.to_numpy()
    features = candidate_features[keep_index]
    table["reference_centroid_cosine"] = [cosine(vector, centroid) for vector in features]
    return table


def aggregate_by_group(table: pd.DataFrame):
    value_columns = [
        "reference_centroid_cosine",
        "seen_mean_accuracy",
        "unseen_mean_accuracy",
        "overall_accuracy",
        "transfer_gap",
    ]
    grouped = (
        table.groupby("group_id", dropna=False)
        .agg(
            prompt_id=("prompt_id", "first"),
            source=("source", "first"),
            variant_count=("prompt_id", "count"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    return grouped


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson(x_rank, y_rank)


def build_summary(group_table, *, top_k: int, random_trials: int, seed: int):
    score = group_table["reference_centroid_cosine"].to_numpy(dtype=np.float64)
    seen = group_table["seen_mean_accuracy"].to_numpy(dtype=np.float64)
    target = group_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)

    return {
        "num_candidate_groups": int(len(group_table)),
        "top_k": int(min(top_k, len(group_table))),
        "activation_cosine_top_k_unseen_accuracy": top_k_mean(score, target, top_k),
        "seen_accuracy_top_k_unseen_accuracy": top_k_mean(seen, target, top_k),
        "random_top_k_unseen_accuracy": random_top_k_mean(
            target,
            top_k,
            trials=random_trials,
            random_state=seed,
        ),
        "activation_cosine_pearson_unseen": pearson(score, target),
        "activation_cosine_spearman_unseen": spearman(score, target),
        "seen_accuracy_pearson_unseen": pearson(seen, target),
        "seen_accuracy_spearman_unseen": spearman(seen, target),
    }


def format_report(summary, reference_slice, reference_top_table, ranked_groups):
    lines = [
        "# Activation Selection Report",
        "",
        "## Reference Slice",
        "",
        f"- Slice type: {reference_slice['slice_type']}",
        f"- Layer: {reference_slice['layer']}",
        f"- Position: {reference_slice['position']}",
        f"- Reference activation ridge R^2: {reference_slice['activation_ridge_r2']:.4f}",
        "",
        "## Reference Top Prompts",
        "",
        *[
            f"- {row.prompt_id}: unseen={row.unseen_mean_accuracy:.4f}, seen={row.seen_mean_accuracy:.4f}"
            for row in reference_top_table.itertuples(index=False)
        ],
        "",
        "## Selection Metrics",
        "",
        f"- Activation-cosine top-k unseen accuracy: {summary['activation_cosine_top_k_unseen_accuracy']:.4f}",
        f"- Seen-accuracy top-k unseen accuracy: {summary['seen_accuracy_top_k_unseen_accuracy']:.4f}",
        f"- Random top-k unseen accuracy: {summary['random_top_k_unseen_accuracy']:.4f}",
        f"- Activation-cosine Spearman with unseen: {summary['activation_cosine_spearman_unseen']:.4f}",
        "",
        "## Activation-Selected Candidates",
        "",
        *[
            f"- {row.prompt_id}: cosine={row.reference_centroid_cosine:.4f}, unseen={row.unseen_mean_accuracy:.4f}, seen={row.seen_mean_accuracy:.4f}"
            for row in ranked_groups.head(summary["top_k"]).itertuples(index=False)
        ],
        "",
    ]
    return "\n".join(lines)


def main():
    args = parse_args()
    reference_config = load_config(args.reference_config)
    candidate_config = load_config(args.candidate_config)

    reference_artifacts = load_run_artifacts(reference_config)
    candidate_artifacts = load_run_artifacts(candidate_config)

    reference_slice = select_reference_slice(reference_artifacts["slice_analysis"])
    layer = reference_slice.get("layer")
    position = reference_slice.get("position")

    reference_features, reference_table, _ = build_slice_features(
        reference_artifacts,
        reference_config,
        layer=layer,
        position=position,
    )
    candidate_features, candidate_table, _ = build_slice_features(
        candidate_artifacts,
        candidate_config,
        layer=layer,
        position=position,
    )

    centroid, reference_top = top_reference_centroid(
        reference_features,
        reference_table,
        top_n=args.top_reference_prompts,
    )
    candidate_ranked = rank_candidates(candidate_features, candidate_table, centroid)
    group_table = aggregate_by_group(candidate_ranked)
    group_ranked = group_table.sort_values(
        ["reference_centroid_cosine", "unseen_mean_accuracy"],
        ascending=[False, False],
    ).reset_index(drop=True)

    top_k = args.top_k if args.top_k is not None else candidate_config["analysis"]["top_k"]
    summary = build_summary(
        group_ranked,
        top_k=top_k,
        random_trials=candidate_config["analysis"]["random_trials"],
        seed=candidate_config.get("seed", 42),
    )
    summary.update(
        {
            "reference_config": args.reference_config,
            "candidate_config": args.candidate_config,
            "reference_slice": reference_slice,
            "reference_top_prompts": reference_top["prompt_id"].tolist(),
        }
    )

    outputs_root = resolve_path(candidate_config["_project_root"], candidate_config["paths"]["outputs_dir"])
    results_dir = ensure_dir(outputs_root / args.output_subdir)
    save_dataframe(candidate_ranked, results_dir / "activation_selection_prompt_table.parquet")
    save_json(results_dir / "activation_selection_prompt_table.json", candidate_ranked.to_dict(orient="records"))
    save_dataframe(group_ranked, results_dir / "activation_selection_group_table.parquet")
    save_json(results_dir / "activation_selection_group_table.json", group_ranked.to_dict(orient="records"))
    save_json(results_dir / "activation_selection_summary.json", summary)
    save_markdown(
        results_dir / "activation_selection_report.md",
        format_report(summary, reference_slice, reference_top, group_ranked),
    )

    print(f"Saved activation-selection outputs to {results_dir}")


if __name__ == "__main__":
    main()
