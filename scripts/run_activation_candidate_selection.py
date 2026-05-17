from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import build_prompt_feature_matrix, merge_prompt_features_with_eval
from src.utils.io import ensure_dir, load_config, load_jsonl, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an activation-performance predictor on an evaluated prompt pool and rank unevaluated candidate prompts."
    )
    parser.add_argument("--reference-config", required=True)
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--output-subdir", default="activation_candidate_selection")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--slice-type", choices=["best", "full"], default="best")
    parser.add_argument("--exclude-base-from-training", action="store_true")
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


def load_artifacts(config):
    project_root = config["_project_root"]
    outputs_root = resolve_path(project_root, config["paths"]["outputs_dir"])
    return {
        "outputs_root": outputs_root,
        "eval_prompt_summary": pd.read_parquet(outputs_root / "eval" / "eval_prompt_summary.parquet")
        if (outputs_root / "eval" / "eval_prompt_summary.parquet").exists()
        else None,
        "activation_summary": pd.read_parquet(outputs_root / "activations" / "activation_summary.parquet"),
        "summary_vectors": np.load(outputs_root / "activations" / "activation_vectors.npz")[
            "summary_vectors"
        ],
    }


def select_best_slice(reference_outputs_root):
    path = reference_outputs_root / "results" / "slice_analysis.parquet"
    if not path.exists():
        return None
    table = pd.read_parquet(path)
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


def build_features(config, artifacts, *, selected_slice, slice_type):
    sliced = artifacts["activation_summary"].copy()
    if slice_type == "best" and selected_slice is not None:
        layer = selected_slice.get("layer")
        position = selected_slice.get("position")
        if layer is not None and not pd.isna(layer):
            sliced = sliced[sliced["layer"] == layer]
        if position is not None and not pd.isna(position):
            sliced = sliced[sliced["position"] == position]

    features, prompt_meta, feature_keys = build_prompt_feature_matrix(
        activation_summary_df=sliced,
        summary_vectors=artifacts["summary_vectors"],
        tasks=config["tasks"]["seen"],
    )
    return features, prompt_meta, feature_keys


def attach_prompt_metadata(config, table):
    project_root = config["_project_root"]
    prompt_path = resolve_path(project_root, config["paths"]["prompts"])
    prompt_records = pd.DataFrame(load_jsonl(prompt_path)).rename(columns={"id": "prompt_id"})
    keep = [
        "prompt_id",
        "taxonomy_role",
        "length_control_role",
        "principle_components_json",
        "paper_title",
        "source_note",
    ]
    keep = [column for column in keep if column in prompt_records.columns]
    if not keep:
        return table
    return table.merge(prompt_records[keep], on="prompt_id", how="left")


def write_selected_prompts(config, ranked, path, top_k):
    project_root = config["_project_root"]
    prompt_path = resolve_path(project_root, config["paths"]["prompts"])
    records = load_jsonl(prompt_path)
    selected_ids = set(ranked.head(top_k)["prompt_id"].tolist())
    selected = [record for record in records if record["id"] in selected_ids]
    selected = sorted(selected, key=lambda record: ranked["prompt_id"].tolist().index(record["id"]))
    with path.open("w", encoding="utf-8") as handle:
        for record in selected:
            copied = dict(record)
            copied["source"] = "activation_selected_candidate"
            copied["selection_source"] = config["paths"]["prompts"]
            handle.write(json.dumps(copied, ensure_ascii=False) + "\n")
    return selected


def main():
    args = parse_args()
    reference_config = load_config(args.reference_config)
    candidate_config = load_config(args.candidate_config)

    reference_artifacts = load_artifacts(reference_config)
    candidate_artifacts = load_artifacts(candidate_config)

    selected_slice = select_best_slice(reference_artifacts["outputs_root"])
    reference_features, reference_meta, reference_feature_keys = build_features(
        reference_config,
        reference_artifacts,
        selected_slice=selected_slice,
        slice_type=args.slice_type,
    )
    candidate_features, candidate_meta, candidate_feature_keys = build_features(
        candidate_config,
        candidate_artifacts,
        selected_slice=selected_slice,
        slice_type=args.slice_type,
    )

    if reference_features.shape[1] != candidate_features.shape[1]:
        raise ValueError(
            "Reference and candidate feature dimensions differ: "
            f"{reference_features.shape[1]} vs {candidate_features.shape[1]}"
        )
    if reference_feature_keys != candidate_feature_keys:
        raise ValueError("Reference and candidate feature keys differ; use matching seen tasks/slices.")

    if reference_artifacts["eval_prompt_summary"] is None:
        raise ValueError("Reference config must have eval/eval_prompt_summary.parquet.")
    reference_features, reference_meta, reference_table = merge_prompt_features_with_eval(
        reference_meta,
        reference_features,
        reference_artifacts["eval_prompt_summary"],
    )
    if args.exclude_base_from_training:
        keep = reference_table["source"] != "base"
        reference_table = reference_table[keep].reset_index(drop=True)
        reference_features = reference_features[keep.to_numpy()]

    y = reference_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ridge", Ridge(alpha=reference_config["analysis"]["ridge_alpha"])),
        ]
    )
    model.fit(reference_features, y)
    train_pred = model.predict(reference_features)
    candidate_pred = model.predict(candidate_features)

    candidate_table = candidate_meta.copy()
    candidate_table["activation_pred_unseen_accuracy"] = candidate_pred
    candidate_table["activation_selection_rank"] = (
        candidate_table["activation_pred_unseen_accuracy"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    candidate_table = attach_prompt_metadata(candidate_config, candidate_table)
    ranked = candidate_table.sort_values(
        ["activation_pred_unseen_accuracy", "prompt_length_words"],
        ascending=[False, True],
    ).reset_index(drop=True)

    out_dir = ensure_dir(candidate_artifacts["outputs_root"] / args.output_subdir)
    save_dataframe(ranked, out_dir / "candidate_activation_scores.parquet")
    save_json(out_dir / "candidate_activation_scores.json", ranked.to_dict(orient="records"))
    selected_path = out_dir / "selected_prompts.jsonl"
    selected = write_selected_prompts(candidate_config, ranked, selected_path, args.top_k)

    summary = {
        "reference_config": args.reference_config,
        "candidate_config": args.candidate_config,
        "num_reference_prompts": int(len(reference_table)),
        "num_candidate_prompts": int(len(candidate_table)),
        "top_k": int(args.top_k),
        "slice_type": args.slice_type,
        "selected_slice": selected_slice,
        "feature_blocks": int(len(reference_feature_keys)),
        "train_pred_pearson_actual": pearson(train_pred, y),
        "train_pred_min": float(np.min(train_pred)),
        "train_pred_max": float(np.max(train_pred)),
        "candidate_pred_min": float(np.min(candidate_pred)),
        "candidate_pred_max": float(np.max(candidate_pred)),
        "selected_prompt_ids": [record["id"] for record in selected],
        "selected_prompts_path": str(selected_path),
    }
    save_json(out_dir / "activation_candidate_selection_summary.json", summary)

    lines = [
        "# Activation Candidate Selection",
        "",
        "## Summary",
        "",
        f"- Reference config: {args.reference_config}",
        f"- Candidate config: {args.candidate_config}",
        f"- Reference prompts: {summary['num_reference_prompts']}",
        f"- Candidate prompts: {summary['num_candidate_prompts']}",
        f"- Top-k selected: {summary['top_k']}",
        f"- Slice mode: {summary['slice_type']}",
        f"- Feature blocks: {summary['feature_blocks']}",
        f"- Train prediction Pearson with actual: {summary['train_pred_pearson_actual']:.4f}",
        "",
        "## Selected Prompts",
        "",
    ]
    for row in ranked.head(args.top_k).itertuples(index=False):
        lines.append(
            "- "
            f"{row.prompt_id}: pred={row.activation_pred_unseen_accuracy:.4f}, "
            f"role={getattr(row, 'taxonomy_role', '')}, "
            f"words={row.prompt_length_words}, "
            f"text={row.prompt_text}"
        )
    lines.extend(
        [
            "",
            "## Top Candidate Ranking",
            "",
        ]
    )
    for row in ranked.head(max(args.top_k, 20)).itertuples(index=False):
        lines.append(
            "- "
            f"{row.prompt_id}: pred={row.activation_pred_unseen_accuracy:.4f}, "
            f"role={getattr(row, 'taxonomy_role', '')}, "
            f"text={row.prompt_text}"
        )
    save_markdown(out_dir / "activation_candidate_selection_report.md", "\n".join(lines))
    print(f"Saved activation candidate selection outputs to {out_dir}")


if __name__ == "__main__":
    main()
