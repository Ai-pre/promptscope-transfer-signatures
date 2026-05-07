from __future__ import annotations

import argparse
import itertools
import json

import numpy as np
import pandas as pd

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import build_prompt_feature_matrix, cosine, merge_prompt_features_with_eval
from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path, save_dataframe, save_json, save_markdown


COMPONENT_NAMES = (
    "concise",
    "careful",
    "format",
    "check",
    "soft_reason",
    "hard_reason",
    "strong_expert",
    "expert",
    "verbose",
    "multiagent",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare component-direction geometry across model families using "
            "dimension-agnostic alignment metrics."
        )
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        required=True,
        help="Principle/design configs for each model family.",
    )
    parser.add_argument("--output-dir", default="outputs/cross_model_direction_alignment")
    parser.add_argument("--slice-type", default="layer_position")
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


def infer_components_from_prompt_id(prompt_id: str):
    prompt_id = str(prompt_id)
    if "plain" in prompt_id or "baseline" in prompt_id:
        return []
    components = []
    for component in COMPONENT_NAMES:
        if component == "expert" and "strong_expert" in prompt_id:
            continue
        if component in prompt_id:
            components.append(component)
    return components


def load_artifacts(config):
    outputs_root = resolve_path(config["_project_root"], config["paths"]["outputs_dir"])
    return {
        "outputs_root": outputs_root,
        "eval_prompt_summary": pd.read_parquet(outputs_root / "eval" / "eval_prompt_summary.parquet"),
        "activation_summary": pd.read_parquet(outputs_root / "activations" / "activation_summary.parquet"),
        "summary_vectors": np.load(outputs_root / "activations" / "activation_vectors.npz")["summary_vectors"],
        "slice_analysis": pd.read_parquet(outputs_root / "results" / "slice_analysis.parquet"),
    }


def select_slice(slice_analysis, slice_type: str):
    table = slice_analysis.copy()
    if slice_type:
        filtered = table[table["slice_type"] == slice_type].copy()
        if not filtered.empty:
            table = filtered
    table = table.sort_values(
        ["activation_ridge_r2", "activation_top_k_unseen_accuracy"],
        ascending=[False, False],
    )
    if table.empty:
        raise ValueError("No slice_analysis rows available.")
    return table.iloc[0].to_dict()


def merge_prompt_metadata(table, config):
    prompts_path = resolve_path(config["_project_root"], config["paths"]["prompts"])
    prompt_meta = pd.DataFrame(load_prompts(prompts_path)).rename(columns={"id": "prompt_id"})
    keep = ["prompt_id", "principle_components_json", "principle_components"]
    available = [column for column in keep if column in prompt_meta.columns]
    merged = table.merge(prompt_meta[available], on="prompt_id", how="left")
    if "principle_components_json" in merged.columns:
        merged["principle_components"] = merged["principle_components_json"].apply(parse_component_list)
    elif "principle_components" in merged.columns:
        merged["principle_components"] = merged["principle_components"].apply(parse_component_list)
    else:
        merged["principle_components"] = merged["prompt_id"].apply(infer_components_from_prompt_id)
    empty_mask = merged["principle_components"].apply(len) == 0
    merged.loc[empty_mask, "principle_components"] = merged.loc[empty_mask, "prompt_id"].apply(
        infer_components_from_prompt_id
    )
    merged["component_key"] = merged["principle_components"].apply(lambda items: tuple(sorted(items)))
    return merged


def build_features_for_config(config_path, slice_type):
    config = load_config(config_path)
    artifacts = load_artifacts(config)
    selected_slice = select_slice(artifacts["slice_analysis"], slice_type)

    activation_summary = artifacts["activation_summary"]
    filtered = activation_summary[activation_summary["task"].isin(config["tasks"]["seen"])].copy()
    if selected_slice["slice_type"] in {"layer", "layer_position"}:
        filtered = filtered[filtered["layer"] == selected_slice["layer"]]
    if selected_slice["slice_type"] in {"position", "layer_position"}:
        filtered = filtered[filtered["position"] == selected_slice["position"]]

    features, prompt_meta, _ = build_prompt_feature_matrix(
        activation_summary_df=filtered,
        summary_vectors=artifacts["summary_vectors"],
        tasks=config["tasks"]["seen"],
    )
    features, prompt_meta, table = merge_prompt_features_with_eval(
        prompt_meta,
        features,
        artifacts["eval_prompt_summary"],
    )
    table = merge_prompt_metadata(table, config)
    non_base = table["source"] != "base"
    return {
        "config_path": config_path,
        "model_name": config["model_name"],
        "selected_slice": selected_slice,
        "features": features[non_base.to_numpy()],
        "table": table[non_base].reset_index(drop=True),
    }


def build_component_directions(features, table):
    index_by_key = {}
    for idx, row in table.iterrows():
        key = tuple(row["component_key"])
        index_by_key.setdefault(key, []).append(idx)

    all_components = sorted({component for components in table["component_key"] for component in components})
    directions = {}
    for component in all_components:
        diffs = []
        for base_key, base_indices in index_by_key.items():
            if component in base_key:
                continue
            target_key = tuple(sorted([*base_key, component]))
            target_indices = index_by_key.get(target_key)
            if not target_indices:
                continue
            base_vector = np.mean(features[base_indices], axis=0)
            target_vector = np.mean(features[target_indices], axis=0)
            diffs.append(target_vector - base_vector)
        if diffs:
            directions[component] = np.mean(np.stack(diffs), axis=0)
    return directions


def centered_linear_cka(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    x = x - x.mean(axis=0, keepdims=True)
    y = y - y.mean(axis=0, keepdims=True)
    xy = np.linalg.norm(x.T @ y, ord="fro") ** 2
    xx = np.linalg.norm(x.T @ x, ord="fro")
    yy = np.linalg.norm(y.T @ y, ord="fro")
    denom = xx * yy
    if denom == 0:
        return float("nan")
    return float(xy / denom)


def pearson(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman(x, y):
    return pearson(pd.Series(x).rank(method="average"), pd.Series(y).rank(method="average"))


def common_prompt_views(left, right):
    common_ids = sorted(set(left["table"]["prompt_id"]) & set(right["table"]["prompt_id"]))
    if not common_ids:
        raise ValueError("No common prompt ids between model runs.")
    left_index = {prompt_id: idx for idx, prompt_id in enumerate(left["table"]["prompt_id"])}
    right_index = {prompt_id: idx for idx, prompt_id in enumerate(right["table"]["prompt_id"])}
    left_indices = [left_index[prompt_id] for prompt_id in common_ids]
    right_indices = [right_index[prompt_id] for prompt_id in common_ids]
    return common_ids, left_indices, right_indices


def direction_score_vector(features, direction):
    return np.asarray([cosine(vector, direction) for vector in features], dtype=np.float64)


def compare_model_pair(left, right, left_directions, right_directions):
    common_ids, left_indices, right_indices = common_prompt_views(left, right)
    left_x = left["features"][left_indices]
    right_x = right["features"][right_indices]
    left_table = left["table"].iloc[left_indices].reset_index(drop=True)
    right_table = right["table"].iloc[right_indices].reset_index(drop=True)

    rows = []
    shared_components = sorted(set(left_directions) & set(right_directions))
    for component in shared_components:
        left_scores = direction_score_vector(left_x, left_directions[component])
        right_scores = direction_score_vector(right_x, right_directions[component])
        rows.append(
            {
                "left_model": left["model_name"],
                "right_model": right["model_name"],
                "component": component,
                "num_common_prompts": len(common_ids),
                "component_score_pearson": pearson(left_scores, right_scores),
                "component_score_spearman": spearman(left_scores, right_scores),
                "direct_direction_cosine": cosine(left_directions[component], right_directions[component])
                if left_directions[component].shape == right_directions[component].shape
                else float("nan"),
            }
        )

    pair_summary = {
        "left_model": left["model_name"],
        "right_model": right["model_name"],
        "num_common_prompts": len(common_ids),
        "feature_cka": centered_linear_cka(left_x, right_x),
        "unseen_accuracy_pearson": pearson(
            left_table["unseen_mean_accuracy"],
            right_table["unseen_mean_accuracy"],
        ),
        "unseen_accuracy_spearman": spearman(
            left_table["unseen_mean_accuracy"],
            right_table["unseen_mean_accuracy"],
        ),
        "seen_accuracy_pearson": pearson(
            left_table["seen_mean_accuracy"],
            right_table["seen_mean_accuracy"],
        ),
        "seen_accuracy_spearman": spearman(
            left_table["seen_mean_accuracy"],
            right_table["seen_mean_accuracy"],
        ),
    }
    return pair_summary, pd.DataFrame(rows)


def format_report(summary_rows, component_rows, model_summaries):
    lines = [
        "# Cross-Model Direction Alignment Report",
        "",
        "## Model Slices",
        "",
    ]
    for row in model_summaries:
        selected = row["selected_slice"]
        lines.append(
            f"- {row['model_name']}: slice={selected['slice_type']}, "
            f"layer={selected.get('layer')}, position={selected.get('position')}"
        )
    lines.extend(["", "## Pairwise Feature Alignment", ""])
    for row in summary_rows:
        lines.append(
            f"- {row['left_model']} vs {row['right_model']}: "
            f"CKA={row['feature_cka']:.4f}, unseen_spearman={row['unseen_accuracy_spearman']:.4f}"
        )
    lines.extend(["", "## Component Score Alignment", ""])
    if component_rows.empty:
        lines.append("- No shared component directions were available.")
    else:
        for row in component_rows.itertuples(index=False):
            lines.append(
                f"- {row.left_model} vs {row.right_model} / {row.component}: "
                f"score_spearman={row.component_score_spearman:.4f}, "
                f"direct_cosine={row.direct_direction_cosine:.4f}"
            )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    runs = [build_features_for_config(config_path, args.slice_type) for config_path in args.configs]
    directions = {
        run["model_name"]: build_component_directions(run["features"], run["table"])
        for run in runs
    }

    pair_summaries = []
    component_tables = []
    for left, right in itertools.combinations(runs, 2):
        pair_summary, component_table = compare_model_pair(
            left,
            right,
            directions[left["model_name"]],
            directions[right["model_name"]],
        )
        pair_summaries.append(pair_summary)
        component_tables.append(component_table)

    component_rows = pd.concat(component_tables, ignore_index=True) if component_tables else pd.DataFrame()
    output_dir = ensure_dir(args.output_dir)
    save_json(
        output_dir / "cross_model_alignment_summary.json",
        {
            "configs": args.configs,
            "model_summaries": [
                {
                    "config_path": run["config_path"],
                    "model_name": run["model_name"],
                    "selected_slice": run["selected_slice"],
                    "num_prompts": int(len(run["table"])),
                    "components": sorted(directions[run["model_name"]]),
                }
                for run in runs
            ],
            "pair_summaries": pair_summaries,
        },
    )
    pair_table = pd.DataFrame(pair_summaries)
    save_dataframe(pair_table, output_dir / "cross_model_pair_summary.parquet")
    save_json(output_dir / "cross_model_pair_summary.json", pair_table.to_dict(orient="records"))
    if not component_rows.empty:
        save_dataframe(component_rows, output_dir / "cross_model_component_alignment.parquet")
        save_json(output_dir / "cross_model_component_alignment.json", component_rows.to_dict(orient="records"))
    save_markdown(
        output_dir / "cross_model_alignment_report.md",
        format_report(
            pair_summaries,
            component_rows,
            [
                {
                    "model_name": run["model_name"],
                    "selected_slice": run["selected_slice"],
                }
                for run in runs
            ],
        ),
    )
    print(f"Saved cross-model direction alignment outputs to {output_dir}")


if __name__ == "__main__":
    main()
