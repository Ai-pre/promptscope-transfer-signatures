from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.analysis.analyzer import (
    build_prompt_feature_matrix,
    merge_prompt_features_with_eval,
    random_top_k_mean,
    top_k_mean,
)
from src.utils.io import ensure_dir, load_config, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run robustness checks for activation-signature transfer prediction."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--slice-type", default="auto", choices=["auto", "layer_position", "position", "layer"])
    parser.add_argument("--layer", type=float, default=None)
    parser.add_argument("--position", default=None)
    parser.add_argument("--bootstrap-trials", type=int, default=1000)
    parser.add_argument("--permutation-trials", type=int, default=1000)
    parser.add_argument("--random-direction-trials", type=int, default=1000)
    parser.add_argument("--output-subdir", default="robustness_results")
    return parser.parse_args()


def load_artifacts(config):
    outputs_root = resolve_path(config["_project_root"], config["paths"]["outputs_dir"])
    return {
        "outputs_root": outputs_root,
        "eval_prompt_summary": pd.read_parquet(outputs_root / "eval" / "eval_prompt_summary.parquet"),
        "activation_summary": pd.read_parquet(outputs_root / "activations" / "activation_summary.parquet"),
        "summary_vectors": np.load(outputs_root / "activations" / "activation_vectors.npz")["summary_vectors"],
        "slice_analysis": pd.read_parquet(outputs_root / "results" / "slice_analysis.parquet"),
    }


def select_slice(slice_analysis, args):
    table = slice_analysis.copy()
    if args.slice_type != "auto":
        table = table[table["slice_type"] == args.slice_type].copy()
    if args.layer is not None:
        table = table[table["layer"] == args.layer].copy()
    if args.position is not None:
        table = table[table["position"] == args.position].copy()
    if table.empty:
        raise ValueError("No slice rows match the requested filters.")
    table = table.sort_values(
        ["activation_ridge_r2", "activation_top_k_unseen_accuracy"],
        ascending=[False, False],
    )
    return table.iloc[0].to_dict()


def build_slice_table(artifacts, config, selected_slice, tasks):
    activation_summary = artifacts["activation_summary"].copy()
    filtered = activation_summary[activation_summary["task"].isin(tasks)].copy()

    if selected_slice["slice_type"] in {"layer", "layer_position"}:
        filtered = filtered[filtered["layer"] == selected_slice["layer"]]
    if selected_slice["slice_type"] in {"position", "layer_position"}:
        filtered = filtered[filtered["position"] == selected_slice["position"]]
    if filtered.empty:
        raise ValueError(f"No activation rows for selected slice: {selected_slice}")

    features, prompt_meta, _ = build_prompt_feature_matrix(
        activation_summary_df=filtered,
        summary_vectors=artifacts["summary_vectors"],
        tasks=tasks,
    )
    features, prompt_meta, table = merge_prompt_features_with_eval(
        prompt_meta,
        features,
        artifacts["eval_prompt_summary"],
    )
    return features, table


def make_splitter(n_samples, n_splits, groups=None):
    if groups is not None:
        unique_groups = pd.Series(groups).nunique(dropna=False)
        actual = min(n_splits, unique_groups)
        if actual >= 2:
            return GroupKFold(n_splits=actual)
    actual = min(n_splits, n_samples)
    if actual < 2:
        return None
    return KFold(n_splits=actual, shuffle=True, random_state=42)


def oof_ridge_predictions(X, y, groups, *, n_splits, alpha):
    splitter = make_splitter(len(X), n_splits, groups)
    if splitter is None:
        return np.full(len(y), np.nan), float("nan")
    preds = np.zeros(len(y), dtype=np.float64)
    split_iter = splitter.split(X, y, groups)
    for train_idx, test_idx in split_iter:
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X[train_idx], y[train_idx])
        preds[test_idx] = model.predict(X[test_idx])
    return preds, float(r2_score(y, preds))


def group_selection_frame(preds, y, seen, group_ids):
    frame = pd.DataFrame(
        {
            "prediction": preds,
            "target": y,
            "seen": seen,
            "group_id": group_ids,
        }
    )
    return (
        frame.groupby("group_id", dropna=False)[["prediction", "target", "seen"]]
        .mean()
        .reset_index()
    )


def percentile_interval(values):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {
        "mean": float(np.nanmean(values)),
        "ci_low": float(np.nanpercentile(values, 2.5)),
        "ci_high": float(np.nanpercentile(values, 97.5)),
    }


def bootstrap_metrics(preds, y, seen, group_ids, *, top_k, trials, seed):
    rng = np.random.default_rng(seed)
    grouped = group_selection_frame(preds, y, seen, group_ids)
    r2_values = []
    activation_topk = []
    seen_topk = []
    for _ in range(trials):
        idx = rng.choice(len(grouped), size=len(grouped), replace=True)
        sample = grouped.iloc[idx]
        if sample["target"].nunique(dropna=False) > 1:
            r2_values.append(r2_score(sample["target"], sample["prediction"]))
        activation_topk.append(
            top_k_mean(
                sample["prediction"].to_numpy(dtype=np.float64),
                sample["target"].to_numpy(dtype=np.float64),
                top_k,
            )
        )
        seen_topk.append(
            top_k_mean(
                sample["seen"].to_numpy(dtype=np.float64),
                sample["target"].to_numpy(dtype=np.float64),
                top_k,
            )
        )
    return {
        "bootstrap_r2": percentile_interval(r2_values),
        "bootstrap_activation_top_k": percentile_interval(activation_topk),
        "bootstrap_seen_top_k": percentile_interval(seen_topk),
    }


def permutation_test(X, y, groups, *, n_splits, alpha, observed_r2, trials, seed):
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(trials):
        shuffled = rng.permutation(y)
        _, score = oof_ridge_predictions(
            X,
            shuffled,
            groups,
            n_splits=n_splits,
            alpha=alpha,
        )
        values.append(score)
    values = np.asarray(values, dtype=np.float64)
    p_value = float((np.sum(values >= observed_r2) + 1) / (len(values) + 1))
    return {
        "permutation_r2_mean": float(np.nanmean(values)),
        "permutation_r2_ci_low": float(np.nanpercentile(values, 2.5)),
        "permutation_r2_ci_high": float(np.nanpercentile(values, 97.5)),
        "permutation_p_value_ge_observed": p_value,
    }


def random_direction_baseline(X, y, seen, group_ids, *, top_k, trials, seed):
    rng = np.random.default_rng(seed)
    values = []
    topk_values = []
    for _ in range(trials):
        direction = rng.normal(size=X.shape[1])
        score = X @ direction
        if np.std(score) > 0 and np.std(y) > 0:
            values.append(float(np.corrcoef(score, y)[0, 1]))
        grouped = group_selection_frame(score, y, seen, group_ids)
        topk_values.append(
            top_k_mean(
                grouped["prediction"].to_numpy(dtype=np.float64),
                grouped["target"].to_numpy(dtype=np.float64),
                top_k,
            )
        )
    return {
        "random_direction_pearson": percentile_interval(values),
        "random_direction_top_k": percentile_interval(topk_values),
    }


def leave_one_seen_task_out(artifacts, config, selected_slice, *, n_splits, alpha, top_k):
    rows = []
    seen_tasks = list(config["tasks"]["seen"])
    for held_out in seen_tasks:
        train_tasks = [task for task in seen_tasks if task != held_out]
        if not train_tasks:
            continue
        X, table = build_slice_table(artifacts, config, selected_slice, train_tasks)
        y = table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
        seen = table["seen_mean_accuracy"].to_numpy(dtype=np.float64)
        groups = table["group_id"].to_numpy()
        preds, r2 = oof_ridge_predictions(
            X,
            y,
            groups,
            n_splits=n_splits,
            alpha=alpha,
        )
        grouped = group_selection_frame(preds, y, seen, groups)
        rows.append(
            {
                "held_out_seen_task": held_out,
                "train_seen_tasks": ",".join(train_tasks),
                "activation_ridge_r2": r2,
                "activation_top_k_unseen_accuracy": top_k_mean(
                    grouped["prediction"].to_numpy(dtype=np.float64),
                    grouped["target"].to_numpy(dtype=np.float64),
                    top_k,
                ),
                "seen_accuracy_top_k_unseen_accuracy": top_k_mean(
                    grouped["seen"].to_numpy(dtype=np.float64),
                    grouped["target"].to_numpy(dtype=np.float64),
                    top_k,
                ),
            }
        )
    return pd.DataFrame(rows)


def format_report(summary, loo_table):
    lines = [
        "# Robustness Analysis Report",
        "",
        "## Selected Slice",
        "",
        f"- Slice type: {summary['selected_slice']['slice_type']}",
        f"- Layer: {summary['selected_slice'].get('layer')}",
        f"- Position: {summary['selected_slice'].get('position')}",
        "",
        "## Main Robustness Metrics",
        "",
        f"- Observed activation R^2: {summary['observed_activation_r2']:.4f}",
        f"- Observed activation top-k unseen: {summary['observed_activation_top_k_unseen_accuracy']:.4f}",
        f"- Observed seen top-k unseen: {summary['observed_seen_top_k_unseen_accuracy']:.4f}",
        f"- Observed random top-k unseen: {summary['observed_random_top_k_unseen_accuracy']:.4f}",
        f"- Permutation p-value: {summary['permutation']['permutation_p_value_ge_observed']:.4f}",
        "",
        "## Bootstrap CIs",
        "",
    ]
    for key, value in summary["bootstrap"].items():
        lines.append(
            f"- {key}: mean={value['mean']:.4f}, 95% CI=[{value['ci_low']:.4f}, {value['ci_high']:.4f}]"
        )
    lines.extend(["", "## Leave-One-Seen-Task-Out", ""])
    for row in loo_table.itertuples(index=False):
        lines.append(
            f"- held_out={row.held_out_seen_task}: R2={row.activation_ridge_r2:.4f}, "
            f"activation_top_k={row.activation_top_k_unseen_accuracy:.4f}, "
            f"seen_top_k={row.seen_accuracy_top_k_unseen_accuracy:.4f}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)
    artifacts = load_artifacts(config)
    selected_slice = select_slice(artifacts["slice_analysis"], args)
    X, table = build_slice_table(artifacts, config, selected_slice, config["tasks"]["seen"])

    y = table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    seen = table["seen_mean_accuracy"].to_numpy(dtype=np.float64)
    groups = table["group_id"].to_numpy()
    n_splits = config["analysis"]["n_splits"]
    alpha = config["analysis"]["ridge_alpha"]
    top_k = config["analysis"]["top_k"]
    seed = config.get("seed", 42)

    preds, observed_r2 = oof_ridge_predictions(
        X,
        y,
        groups,
        n_splits=n_splits,
        alpha=alpha,
    )
    grouped = group_selection_frame(preds, y, seen, groups)
    observed_activation_topk = top_k_mean(
        grouped["prediction"].to_numpy(dtype=np.float64),
        grouped["target"].to_numpy(dtype=np.float64),
        top_k,
    )
    observed_seen_topk = top_k_mean(
        grouped["seen"].to_numpy(dtype=np.float64),
        grouped["target"].to_numpy(dtype=np.float64),
        top_k,
    )
    observed_random_topk = random_top_k_mean(
        grouped["target"].to_numpy(dtype=np.float64),
        top_k,
        trials=config["analysis"]["random_trials"],
        random_state=seed,
    )

    bootstrap = bootstrap_metrics(
        preds,
        y,
        seen,
        groups,
        top_k=top_k,
        trials=args.bootstrap_trials,
        seed=seed,
    )
    permutation = permutation_test(
        X,
        y,
        groups,
        n_splits=n_splits,
        alpha=alpha,
        observed_r2=observed_r2,
        trials=args.permutation_trials,
        seed=seed + 1,
    )
    random_direction = random_direction_baseline(
        X,
        y,
        seen,
        groups,
        top_k=top_k,
        trials=args.random_direction_trials,
        seed=seed + 2,
    )
    loo_table = leave_one_seen_task_out(
        artifacts,
        config,
        selected_slice,
        n_splits=n_splits,
        alpha=alpha,
        top_k=top_k,
    )

    summary = {
        "config": args.config,
        "selected_slice": selected_slice,
        "observed_activation_r2": observed_r2,
        "observed_activation_top_k_unseen_accuracy": observed_activation_topk,
        "observed_seen_top_k_unseen_accuracy": observed_seen_topk,
        "observed_random_top_k_unseen_accuracy": observed_random_topk,
        "bootstrap": bootstrap,
        "permutation": permutation,
        "random_direction": random_direction,
    }

    results_dir = ensure_dir(artifacts["outputs_root"] / args.output_subdir)
    save_json(results_dir / "robustness_summary.json", summary)
    save_dataframe(loo_table, results_dir / "leave_one_seen_task_out.parquet")
    save_json(results_dir / "leave_one_seen_task_out.json", loo_table.to_dict(orient="records"))
    save_markdown(results_dir / "robustness_report.md", format_report(summary, loo_table))
    print(f"Saved robustness analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()
