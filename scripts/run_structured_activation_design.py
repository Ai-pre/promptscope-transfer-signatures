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
    random_top_k_mean,
    top_k_mean,
)
from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Score activation-informed designed prompts with component directions "
            "such as concise_direction and format_direction."
        )
    )
    parser.add_argument("--reference-config", default="configs/config.gpu_general_mixed_clean.yaml")
    parser.add_argument("--candidate-config", default="configs/config.gpu_principle_boundary.yaml")
    parser.add_argument("--top-reference-prompts", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--output-subdir", default="structured_activation_design")
    parser.add_argument(
        "--target-components",
        nargs="+",
        default=["concise", "careful", "format"],
        help=(
            "Component directions used for the main structured score. "
            "Defaults to the lightweight boundary-setting hypothesis."
        ),
    )
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
        "slice_analysis": pd.read_parquet(results_dir / "slice_analysis.parquet"),
    }


def select_reference_slice(slice_analysis: pd.DataFrame):
    if slice_analysis.empty:
        raise ValueError("Reference slice_analysis is empty. Run reference analysis first.")

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


def merge_candidate_prompt_metadata(candidate_table, candidate_config):
    prompts_path = resolve_path(candidate_config["_project_root"], candidate_config["paths"]["prompts"])
    prompt_meta = pd.DataFrame(load_prompts(prompts_path)).rename(columns={"id": "prompt_id"})
    keep_columns = [
        "prompt_id",
        "principle_components_json",
        "principle_family",
        "complexity_level",
        "hypothesis_role",
        "contrast_group",
        "source_note",
    ]
    available = [column for column in keep_columns if column in prompt_meta.columns]
    merged = candidate_table.merge(prompt_meta[available], on="prompt_id", how="left")
    merged["principle_components"] = merged["principle_components_json"].apply(parse_component_list)
    merged["component_key"] = merged["principle_components"].apply(lambda items: tuple(sorted(items)))
    return merged


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


def build_component_directions(candidate_features, candidate_table):
    # The neutral base prompt is useful for delta_h extraction, but component
    # directions should be estimated only from controlled designed prompts.
    valid_mask = candidate_table["source"] != "base"
    candidate_features = candidate_features[valid_mask.to_numpy()]
    candidate_table = candidate_table[valid_mask].reset_index(drop=True)

    index_by_key = {}
    for idx, row in candidate_table.iterrows():
        key = tuple(row["component_key"])
        index_by_key.setdefault(key, []).append(idx)

    all_components = sorted({component for components in candidate_table["component_key"] for component in components})
    rows = []
    directions = {}

    for component in all_components:
        diffs = []
        pair_labels = []
        for base_key, base_indices in index_by_key.items():
            if component in base_key:
                continue
            target_key = tuple(sorted([*base_key, component]))
            target_indices = index_by_key.get(target_key)
            if not target_indices:
                continue

            base_vector = np.mean(candidate_features[base_indices], axis=0)
            target_vector = np.mean(candidate_features[target_indices], axis=0)
            diffs.append(target_vector - base_vector)
            pair_labels.append(f"{'+'.join(target_key) or 'plain'} - {'+'.join(base_key) or 'plain'}")

        if not diffs:
            continue

        direction = np.mean(np.stack(diffs), axis=0)
        directions[component] = direction
        rows.append(
            {
                "component": component,
                "num_minimal_pairs": len(diffs),
                "direction_norm": float(np.linalg.norm(direction)),
                "minimal_pairs": "; ".join(pair_labels),
            }
        )

    return directions, pd.DataFrame(rows)


def add_direction_scores(
    candidate_features,
    candidate_table,
    component_directions,
    reference_centroid,
    target_components,
):
    scored = candidate_table[candidate_table["source"] != "base"].copy().reset_index(drop=True)
    keep_index = candidate_table[candidate_table["source"] != "base"].index.to_numpy()
    features = candidate_features[keep_index]

    components = sorted(component_directions)
    for component in components:
        scored[f"{component}_direction_cosine"] = [
            cosine(vector, component_directions[component]) for vector in features
        ]

    if components:
        score_columns = [f"{component}_direction_cosine" for component in components]
        target_score_columns = [
            f"{component}_direction_cosine"
            for component in target_components
            if component in component_directions
        ]
        if target_score_columns:
            scored["structured_component_score"] = scored[target_score_columns].mean(axis=1)
        else:
            scored["structured_component_score"] = scored[score_columns].mean(axis=1)
        scored["own_component_score"] = [
            float(np.mean([row[f"{component}_direction_cosine"] for component in row["principle_components"] if component in components]))
            if any(component in components for component in row["principle_components"])
            else float("nan")
            for _, row in scored.iterrows()
        ]
    else:
        scored["structured_component_score"] = np.nan
        scored["own_component_score"] = np.nan

    scored["reference_centroid_cosine"] = [cosine(vector, reference_centroid) for vector in features]
    return scored


def aggregate_by_group(table: pd.DataFrame):
    value_columns = [
        "structured_component_score",
        "own_component_score",
        "reference_centroid_cosine",
        "seen_mean_accuracy",
        "unseen_mean_accuracy",
        "overall_accuracy",
        "transfer_gap",
    ]
    direction_columns = [column for column in table.columns if column.endswith("_direction_cosine")]
    value_columns.extend(direction_columns)

    grouped = (
        table.groupby("group_id", dropna=False)
        .agg(
            prompt_id=("prompt_id", "first"),
            source=("source", "first"),
            variant_count=("prompt_id", "count"),
            principle_components_json=("principle_components_json", "first"),
            complexity_level=("complexity_level", "first"),
            **{column: (column, "mean") for column in value_columns},
        )
        .reset_index()
    )
    return grouped


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.nanstd(x) == 0 or np.nanstd(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    x_rank = pd.Series(x).rank(method="average").to_numpy(dtype=np.float64)
    y_rank = pd.Series(y).rank(method="average").to_numpy(dtype=np.float64)
    return pearson(x_rank, y_rank)


def build_summary(group_table, *, top_k: int, random_trials: int, seed: int):
    target = group_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    seen = group_table["seen_mean_accuracy"].to_numpy(dtype=np.float64)
    structured = group_table["structured_component_score"].to_numpy(dtype=np.float64)
    centroid = group_table["reference_centroid_cosine"].to_numpy(dtype=np.float64)

    return {
        "num_candidate_groups": int(len(group_table)),
        "top_k": int(min(top_k, len(group_table))),
        "structured_component_top_k_unseen_accuracy": top_k_mean(structured, target, top_k),
        "reference_centroid_top_k_unseen_accuracy": top_k_mean(centroid, target, top_k),
        "seen_accuracy_top_k_unseen_accuracy": top_k_mean(seen, target, top_k),
        "random_top_k_unseen_accuracy": random_top_k_mean(
            target,
            top_k,
            trials=random_trials,
            random_state=seed,
        ),
        "structured_component_pearson_unseen": pearson(structured, target),
        "structured_component_spearman_unseen": spearman(structured, target),
        "reference_centroid_pearson_unseen": pearson(centroid, target),
        "reference_centroid_spearman_unseen": spearman(centroid, target),
        "seen_accuracy_pearson_unseen": pearson(seen, target),
        "seen_accuracy_spearman_unseen": spearman(seen, target),
    }


def format_report(summary, reference_slice, reference_top, component_direction_table, ranked_groups):
    lines = [
        "# Structured Activation Design Report",
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
            for row in reference_top.itertuples(index=False)
        ],
        "",
        "## Component Directions",
        "",
        *[
            f"- {row.component}: pairs={row.num_minimal_pairs}, norm={row.direction_norm:.4f}, examples={row.minimal_pairs}"
            for row in component_direction_table.itertuples(index=False)
        ],
        "",
        "## Selection Metrics",
        "",
        f"- Structured-component top-k unseen accuracy: {summary['structured_component_top_k_unseen_accuracy']:.4f}",
        f"- Reference-centroid top-k unseen accuracy: {summary['reference_centroid_top_k_unseen_accuracy']:.4f}",
        f"- Seen-accuracy top-k unseen accuracy: {summary['seen_accuracy_top_k_unseen_accuracy']:.4f}",
        f"- Random top-k unseen accuracy: {summary['random_top_k_unseen_accuracy']:.4f}",
        f"- Structured-component Spearman with unseen: {summary['structured_component_spearman_unseen']:.4f}",
        f"- Target components: {', '.join(summary['target_components'])}",
        "",
        "## Structured-Selected Candidates",
        "",
        *[
            f"- {row.prompt_id}: score={row.structured_component_score:.4f}, unseen={row.unseen_mean_accuracy:.4f}, seen={row.seen_mean_accuracy:.4f}"
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
    candidate_table = merge_candidate_prompt_metadata(candidate_table, candidate_config)

    centroid, reference_top = top_reference_centroid(
        reference_features,
        reference_table,
        top_n=args.top_reference_prompts,
    )
    component_directions, component_direction_table = build_component_directions(
        candidate_features,
        candidate_table,
    )
    if not component_directions:
        raise ValueError("No component directions could be built from the candidate prompt set.")

    scored_prompts = add_direction_scores(
        candidate_features,
        candidate_table,
        component_directions,
        centroid,
        args.target_components,
    )
    group_table = aggregate_by_group(scored_prompts)
    group_ranked = group_table.sort_values(
        ["structured_component_score", "unseen_mean_accuracy"],
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
            "target_components": args.target_components,
            "component_directions": component_direction_table.to_dict(orient="records"),
        }
    )

    outputs_root = resolve_path(candidate_config["_project_root"], candidate_config["paths"]["outputs_dir"])
    results_dir = ensure_dir(outputs_root / args.output_subdir)
    save_dataframe(scored_prompts, results_dir / "structured_activation_prompt_table.parquet")
    save_json(results_dir / "structured_activation_prompt_table.json", scored_prompts.to_dict(orient="records"))
    save_dataframe(group_ranked, results_dir / "structured_activation_group_table.parquet")
    save_json(results_dir / "structured_activation_group_table.json", group_ranked.to_dict(orient="records"))
    save_dataframe(component_direction_table, results_dir / "component_direction_table.parquet")
    save_json(results_dir / "component_direction_table.json", component_direction_table.to_dict(orient="records"))
    save_json(results_dir / "structured_activation_summary.json", summary)
    save_markdown(
        results_dir / "structured_activation_report.md",
        format_report(summary, reference_slice, reference_top, component_direction_table, group_ranked),
    )

    print(f"Saved structured activation design outputs to {results_dir}")


if __name__ == "__main__":
    main()
