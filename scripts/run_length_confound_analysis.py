from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path, save_dataframe, save_json, save_markdown


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run length/format confound checks for controlled prompt pools. "
            "This estimates whether prompt components still predict unseen accuracy "
            "after controlling for generated length, prompt length, and final-answer markers."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-subdir", default="length_confound_results")
    parser.add_argument("--bootstrap-trials", type=int, default=2000)
    parser.add_argument("--permutation-trials", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def parse_component_list(value):
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    return [str(value)]


def count_words(text):
    return len(str(text).strip().split())


def has_final_answer_marker(text):
    return "final answer" in str(text).lower()


def load_prompt_metadata(config):
    prompts_path = resolve_path(config["_project_root"], config["paths"]["prompts"])
    prompts = pd.DataFrame(load_prompts(prompts_path)).rename(columns={"id": "prompt_id"})
    keep = [
        "prompt_id",
        "group_id",
        "source",
        "text",
        "prompt_length_words",
        "principle_components_json",
        "length_control_role",
        "taxonomy_role",
        "source_note",
        "paper_title",
    ]
    available = [column for column in keep if column in prompts.columns]
    prompts = prompts[available].copy()
    if "principle_components_json" not in prompts.columns:
        prompts["principle_components_json"] = "[]"
    prompts["principle_components"] = prompts["principle_components_json"].apply(parse_component_list)
    if "length_control_role" not in prompts.columns:
        prompts["length_control_role"] = prompts.get("taxonomy_role", "unknown")
    prompts["prompt_length_words"] = prompts.get("prompt_length_words", prompts["text"].apply(count_words))
    return prompts


def add_sample_features(eval_results):
    frame = eval_results.copy()
    frame["prediction_word_count"] = frame["prediction"].apply(count_words)
    frame["has_final_answer_marker"] = frame["prediction"].apply(has_final_answer_marker).astype(float)
    return frame


def build_prompt_task_table(eval_results, prompt_meta):
    frame = add_sample_features(eval_results)
    grouped = (
        frame.groupby(["prompt_id", "group_id", "source", "task", "split"], dropna=False)
        .agg(
            accuracy=("correct", "mean"),
            mean_prediction_words=("prediction_word_count", "mean"),
            median_prediction_words=("prediction_word_count", "median"),
            final_answer_marker_rate=("has_final_answer_marker", "mean"),
            num_samples=("correct", "count"),
        )
        .reset_index()
    )
    grouped = grouped.merge(
        prompt_meta,
        on=["prompt_id", "group_id", "source"],
        how="left",
    )
    return grouped


def build_prompt_table(prompt_task_table):
    rows = []
    for key, group in prompt_task_table.groupby(["prompt_id", "group_id", "source"], dropna=False):
        row = {column: value for column, value in zip(["prompt_id", "group_id", "source"], key)}
        seen = group[group["split"] == "seen"]
        unseen = group[group["split"] == "unseen"]
        row["seen_mean_accuracy"] = float(seen["accuracy"].mean()) if not seen.empty else float("nan")
        row["unseen_mean_accuracy"] = float(unseen["accuracy"].mean()) if not unseen.empty else float("nan")
        row["overall_accuracy"] = float(group["accuracy"].mean())
        row["mean_prediction_words"] = float(group["mean_prediction_words"].mean())
        row["final_answer_marker_rate"] = float(group["final_answer_marker_rate"].mean())
        representative = group.iloc[0]
        for column in [
            "text",
            "prompt_length_words",
            "principle_components_json",
            "length_control_role",
            "taxonomy_role",
            "source_note",
            "paper_title",
        ]:
            if column in group.columns:
                row[column] = representative.get(column)
        rows.append(row)
    prompt_table = pd.DataFrame(rows)
    prompt_table["principle_components"] = prompt_table["principle_components_json"].apply(parse_component_list)
    return prompt_table


def component_names(prompt_table):
    names = set()
    for components in prompt_table["principle_components"]:
        names.update(parse_component_list(components))
    return sorted(names)


def add_component_columns(frame, components):
    updated = frame.copy()
    parsed = updated["principle_components"].apply(parse_component_list)
    for component in components:
        updated[f"component__{component}"] = parsed.apply(lambda items: float(component in items))
    return updated


def covariate_columns(frame):
    candidates = [
        "mean_prediction_words",
        "prompt_length_words",
        "final_answer_marker_rate",
    ]
    return [column for column in candidates if column in frame.columns and frame[column].notna().any()]


def design_matrix(frame, columns, *, add_task_dummies=False):
    pieces = []
    names = []
    numeric = frame[columns].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if columns:
        scaled = StandardScaler().fit_transform(numeric)
        pieces.append(scaled)
        names.extend(columns)
    if add_task_dummies and "task" in frame.columns:
        dummies = pd.get_dummies(frame["task"].astype(str), prefix="task", drop_first=True, dtype=float)
        if not dummies.empty:
            pieces.append(dummies.to_numpy(dtype=np.float64))
            names.extend(dummies.columns.tolist())
    if not pieces:
        return np.ones((len(frame), 1), dtype=np.float64), ["intercept_only"]
    return np.concatenate(pieces, axis=1), names


def residualize(values, covariates):
    y = np.asarray(values, dtype=np.float64)
    valid = ~(np.isnan(y) | np.any(np.isnan(covariates), axis=1))
    residuals = np.full(len(y), np.nan, dtype=np.float64)
    if valid.sum() < 2:
        return residuals
    model = LinearRegression()
    model.fit(covariates[valid], y[valid])
    residuals[valid] = y[valid] - model.predict(covariates[valid])
    return residuals


def safe_corr(left, right):
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = ~(np.isnan(x) | np.isnan(y))
    x = x[valid]
    y = y[valid]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_interval(values, rng, trials):
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    samples = []
    for _ in range(trials):
        idx = rng.choice(len(values), size=len(values), replace=True)
        samples.append(float(np.mean(values[idx])))
    return {
        "mean": float(np.mean(values)),
        "ci_low": float(np.percentile(samples, 2.5)),
        "ci_high": float(np.percentile(samples, 97.5)),
    }


def component_adjusted_effects(prompt_table, *, bootstrap_trials, permutation_trials, seed):
    rng = np.random.default_rng(seed)
    components = component_names(prompt_table)
    table = add_component_columns(prompt_table, components)
    covariates, cov_names = design_matrix(table, covariate_columns(table), add_task_dummies=False)
    y = table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    y_resid = residualize(y, covariates)
    rows = []

    for component in components:
        col = f"component__{component}"
        x = table[col].to_numpy(dtype=np.float64)
        if np.unique(x[~np.isnan(x)]).size < 2:
            continue
        x_resid = residualize(x, covariates)
        present = y_resid[x == 1.0]
        absent = y_resid[x == 0.0]
        raw_present = y[x == 1.0]
        raw_absent = y[x == 0.0]
        adjusted_delta = float(np.nanmean(present) - np.nanmean(absent))
        raw_delta = float(np.nanmean(raw_present) - np.nanmean(raw_absent))

        boot_deltas = []
        for _ in range(bootstrap_trials):
            idx = rng.choice(len(table), size=len(table), replace=True)
            sample_x = x[idx]
            sample_y = y_resid[idx]
            if np.unique(sample_x).size < 2:
                continue
            boot_deltas.append(float(np.nanmean(sample_y[sample_x == 1.0]) - np.nanmean(sample_y[sample_x == 0.0])))

        perm_deltas = []
        for _ in range(permutation_trials):
            shuffled = rng.permutation(x)
            perm_deltas.append(
                float(np.nanmean(y_resid[shuffled == 1.0]) - np.nanmean(y_resid[shuffled == 0.0]))
            )
        perm_deltas = np.asarray(perm_deltas, dtype=np.float64)
        if adjusted_delta >= 0:
            p_value = float((np.sum(perm_deltas >= adjusted_delta) + 1) / (len(perm_deltas) + 1))
        else:
            p_value = float((np.sum(perm_deltas <= adjusted_delta) + 1) / (len(perm_deltas) + 1))

        rows.append(
            {
                "component": component,
                "present_count": int(np.sum(x == 1.0)),
                "absent_count": int(np.sum(x == 0.0)),
                "raw_delta_unseen_accuracy": raw_delta,
                "adjusted_delta_unseen_accuracy": adjusted_delta,
                "adjusted_delta_ci_low": float(np.nanpercentile(boot_deltas, 2.5)) if boot_deltas else float("nan"),
                "adjusted_delta_ci_high": float(np.nanpercentile(boot_deltas, 97.5)) if boot_deltas else float("nan"),
                "partial_correlation": safe_corr(x_resid, y_resid),
                "permutation_p_value_one_sided": p_value,
                "covariates": cov_names,
            }
        )
    return pd.DataFrame(rows).sort_values("adjusted_delta_unseen_accuracy", ascending=False)


def task_level_adjusted_effects(prompt_task_table, *, bootstrap_trials, seed):
    rng = np.random.default_rng(seed + 101)
    unseen = prompt_task_table[prompt_task_table["split"] == "unseen"].copy()
    if unseen.empty:
        return pd.DataFrame()
    unseen["principle_components"] = unseen["principle_components_json"].apply(parse_component_list)
    components = component_names(unseen)
    unseen = add_component_columns(unseen, components)
    covariates = covariate_columns(unseen)
    base_columns = list(covariates)
    X_base, cov_names = design_matrix(unseen, base_columns, add_task_dummies=True)
    y = unseen["accuracy"].to_numpy(dtype=np.float64)
    rows = []
    for component in components:
        col = f"component__{component}"
        if unseen[col].nunique(dropna=True) < 2:
            continue
        x_component = unseen[[col]].to_numpy(dtype=np.float64)
        X = np.concatenate([x_component, X_base], axis=1)
        model = LinearRegression()
        model.fit(X, y)
        coefficient = float(model.coef_[0])

        boot_coefs = []
        for _ in range(bootstrap_trials):
            idx = rng.choice(len(unseen), size=len(unseen), replace=True)
            if np.unique(x_component[idx]).size < 2:
                continue
            boot_model = LinearRegression()
            boot_model.fit(X[idx], y[idx])
            boot_coefs.append(float(boot_model.coef_[0]))

        rows.append(
            {
                "component": component,
                "task_level_adjusted_coefficient": coefficient,
                "coefficient_ci_low": float(np.nanpercentile(boot_coefs, 2.5)) if boot_coefs else float("nan"),
                "coefficient_ci_high": float(np.nanpercentile(boot_coefs, 97.5)) if boot_coefs else float("nan"),
                "num_rows": int(len(unseen)),
                "covariates": ["component"] + cov_names,
            }
        )
    return pd.DataFrame(rows).sort_values("task_level_adjusted_coefficient", ascending=False)


def role_adjusted_effects(prompt_table):
    covariates, _ = design_matrix(prompt_table, covariate_columns(prompt_table), add_task_dummies=False)
    y = prompt_table["unseen_mean_accuracy"].to_numpy(dtype=np.float64)
    residual = residualize(y, covariates)
    adjusted = prompt_table.copy()
    adjusted["adjusted_unseen_accuracy_residual"] = residual
    return (
        adjusted.groupby("length_control_role", dropna=False)
        .agg(
            prompt_count=("prompt_id", "count"),
            raw_unseen_mean_accuracy=("unseen_mean_accuracy", "mean"),
            adjusted_unseen_accuracy_residual=("adjusted_unseen_accuracy_residual", "mean"),
            mean_prediction_words=("mean_prediction_words", "mean"),
            final_answer_marker_rate=("final_answer_marker_rate", "mean"),
        )
        .reset_index()
        .sort_values("adjusted_unseen_accuracy_residual", ascending=False)
    )


def build_summary(prompt_table, component_table, task_component_table, role_table):
    length_corr = safe_corr(prompt_table["mean_prediction_words"], prompt_table["unseen_mean_accuracy"])
    marker_corr = safe_corr(prompt_table["final_answer_marker_rate"], prompt_table["unseen_mean_accuracy"])
    strongest = component_table.iloc[0].to_dict() if not component_table.empty else {}
    strongest_task = task_component_table.iloc[0].to_dict() if not task_component_table.empty else {}
    return {
        "num_prompts": int(len(prompt_table)),
        "prediction_length_pearson_unseen": length_corr,
        "final_answer_marker_pearson_unseen": marker_corr,
        "strongest_prompt_level_adjusted_component": strongest,
        "strongest_task_level_adjusted_component": strongest_task,
        "role_adjusted_effects": role_table.to_dict(orient="records"),
        "component_adjusted_effects": component_table.to_dict(orient="records"),
        "task_level_component_adjusted_effects": task_component_table.to_dict(orient="records"),
        "interpretation_note": (
            "This is a sensitivity analysis, not a causal adjustment. Generated length "
            "and marker use are post-treatment variables, so adjusted effects should be "
            "read as evidence about whether component effects are reducible to these "
            "observable confounds."
        ),
    }


def format_report(summary, component_table, task_component_table, role_table):
    lines = [
        "# Length / Decoding Confound Analysis",
        "",
        "## Summary",
        "",
        f"- Number of prompts: {summary['num_prompts']}",
        f"- Prediction length Pearson with unseen accuracy: {summary['prediction_length_pearson_unseen']:.4f}",
        f"- FINAL ANSWER marker Pearson with unseen accuracy: {summary['final_answer_marker_pearson_unseen']:.4f}",
        "",
        "This analysis controls for generated answer length, prompt length, and final-answer marker rate.",
        "Because generated length and marker use are post-treatment variables, treat this as a sensitivity check rather than a causal adjustment.",
        "",
        "## Prompt-Level Adjusted Component Effects",
        "",
    ]
    if component_table.empty:
        lines.append("- No component effects could be estimated.")
    else:
        for row in component_table.itertuples(index=False):
            lines.append(
                f"- {row.component}: raw_delta={row.raw_delta_unseen_accuracy:.4f}, "
                f"adjusted_delta={row.adjusted_delta_unseen_accuracy:.4f}, "
                f"CI=[{row.adjusted_delta_ci_low:.4f}, {row.adjusted_delta_ci_high:.4f}], "
                f"partial_r={row.partial_correlation:.4f}, p={row.permutation_p_value_one_sided:.4f}"
            )

    lines.extend(["", "## Task-Level Adjusted Component Coefficients", ""])
    if task_component_table.empty:
        lines.append("- No task-level component effects could be estimated.")
    else:
        for row in task_component_table.itertuples(index=False):
            lines.append(
                f"- {row.component}: coef={row.task_level_adjusted_coefficient:.4f}, "
                f"CI=[{row.coefficient_ci_low:.4f}, {row.coefficient_ci_high:.4f}]"
            )

    lines.extend(["", "## Role-Level Adjusted Residuals", ""])
    for row in role_table.itertuples(index=False):
        lines.append(
            f"- {row.length_control_role}: raw_unseen={row.raw_unseen_mean_accuracy:.4f}, "
            f"adjusted_residual={row.adjusted_unseen_accuracy_residual:.4f}, "
            f"words={row.mean_prediction_words:.2f}, marker={row.final_answer_marker_rate:.3f}"
        )
    lines.append("")
    return "\n".join(lines)


def main():
    args = parse_args()
    config = load_config(args.config)
    outputs_root = resolve_path(config["_project_root"], config["paths"]["outputs_dir"])
    eval_dir = outputs_root / "eval"
    results_dir = ensure_dir(outputs_root / args.output_subdir)

    eval_results = pd.read_parquet(eval_dir / "eval_results.parquet")
    prompt_meta = load_prompt_metadata(config)
    prompt_task_table = build_prompt_task_table(eval_results, prompt_meta)
    prompt_table = build_prompt_table(prompt_task_table)
    component_table = component_adjusted_effects(
        prompt_table,
        bootstrap_trials=args.bootstrap_trials,
        permutation_trials=args.permutation_trials,
        seed=args.seed,
    )
    task_component_table = task_level_adjusted_effects(
        prompt_task_table,
        bootstrap_trials=args.bootstrap_trials,
        seed=args.seed,
    )
    role_table = role_adjusted_effects(prompt_table)
    summary = build_summary(prompt_table, component_table, task_component_table, role_table)
    summary.update({"config": args.config, "outputs_dir": str(outputs_root)})

    save_dataframe(prompt_task_table, results_dir / "length_confound_task_table.parquet")
    save_json(results_dir / "length_confound_task_table.json", prompt_task_table.to_dict(orient="records"))
    save_dataframe(prompt_table, results_dir / "length_confound_prompt_table.parquet")
    save_json(results_dir / "length_confound_prompt_table.json", prompt_table.to_dict(orient="records"))
    save_dataframe(component_table, results_dir / "length_confound_component_effects.parquet")
    save_json(results_dir / "length_confound_component_effects.json", component_table.to_dict(orient="records"))
    save_dataframe(task_component_table, results_dir / "length_confound_task_component_effects.parquet")
    save_json(
        results_dir / "length_confound_task_component_effects.json",
        task_component_table.to_dict(orient="records"),
    )
    save_dataframe(role_table, results_dir / "length_confound_role_effects.parquet")
    save_json(results_dir / "length_confound_role_effects.json", role_table.to_dict(orient="records"))
    save_json(results_dir / "length_confound_summary.json", summary)
    save_markdown(
        results_dir / "length_confound_report.md",
        format_report(summary, component_table, task_component_table, role_table),
    )
    print(f"Saved length/decoding confound analysis outputs to {results_dir}")


if __name__ == "__main__":
    main()
