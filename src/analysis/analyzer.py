from __future__ import annotations

import json
import re

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


EPS = 1e-12


def cosine(a, b):
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + EPS
    return float(np.dot(a, b) / denom)


def compute_similarity_matrix(vectors):
    n = len(vectors)
    sim = np.zeros((n, n), dtype=np.float32)

    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine(vectors[i], vectors[j])

    return sim


def tokenize_prompt_text(text: str):
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def lexical_similarity(text: str, reference: str):
    left = tokenize_prompt_text(text)
    right = tokenize_prompt_text(reference)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def build_prompt_feature_matrix(activation_summary_df, summary_vectors, tasks=None):
    df = activation_summary_df.copy().reset_index(drop=True)
    df["vector_row"] = np.arange(len(df))

    if tasks is not None:
        df = df[df["task"].isin(tasks)].copy()

    if df.empty:
        raise ValueError("No activation rows available for the selected tasks.")

    key_columns = ["task", "layer", "position"]
    prompt_columns = [
        "prompt_id",
        "group_id",
        "variant",
        "source",
        "prompt_text",
        "prompt_length_chars",
        "prompt_length_words",
    ]

    feature_keys = [
        tuple(row)
        for row in (
            df[key_columns]
            .drop_duplicates()
            .sort_values(key_columns)
            .itertuples(index=False, name=None)
        )
    ]
    vector_dim = int(summary_vectors.shape[1])
    zero_vector = np.zeros(vector_dim, dtype=np.float32)

    feature_rows = []
    meta_rows = []
    for _, group in df.groupby(prompt_columns, dropna=False):
        group = group.copy()
        chunks = []
        for task_name, layer, position in feature_keys:
            matched = group[
                (group["task"] == task_name)
                & (group["layer"] == layer)
                & (group["position"] == position)
            ]
            if matched.empty:
                chunks.append(zero_vector)
                continue
            row_index = int(matched.iloc[0]["vector_row"])
            chunks.append(summary_vectors[row_index])
        feature_rows.append(np.concatenate(chunks, axis=0))
        meta_rows.append(group.iloc[0][prompt_columns].to_dict())

    return np.vstack(feature_rows), pd.DataFrame(meta_rows), feature_keys


def build_pairwise_similarity_table(prompt_meta, similarity_matrix):
    rows = []
    for i in range(len(prompt_meta)):
        for j in range(i + 1, len(prompt_meta)):
            rows.append(
                {
                    "prompt_id_a": prompt_meta.iloc[i]["prompt_id"],
                    "prompt_id_b": prompt_meta.iloc[j]["prompt_id"],
                    "group_id_a": prompt_meta.iloc[i]["group_id"],
                    "group_id_b": prompt_meta.iloc[j]["group_id"],
                    "variant_a": prompt_meta.iloc[i]["variant"],
                    "variant_b": prompt_meta.iloc[j]["variant"],
                    "cosine_similarity": float(similarity_matrix[i, j]),
                }
            )
    return pd.DataFrame(rows)


def compute_paraphrase_stability(prompt_meta, prompt_features, eval_prompt_summary=None):
    rows = []
    prompt_index = {row["prompt_id"]: idx for idx, row in prompt_meta.iterrows()}

    for group_id, group in prompt_meta.groupby("group_id", dropna=False):
        if len(group) < 2:
            continue

        original_rows = group[group["variant"] == "original"]
        reference = original_rows.iloc[0] if not original_rows.empty else group.iloc[0]
        ref_index = prompt_index[reference["prompt_id"]]

        for _, row in group.iterrows():
            if row["prompt_id"] == reference["prompt_id"]:
                continue
            target_index = prompt_index[row["prompt_id"]]
            rows.append(
                {
                    "group_id": group_id,
                    "original_prompt_id": reference["prompt_id"],
                    "paraphrase_prompt_id": row["prompt_id"],
                    "original_variant": reference["variant"],
                    "paraphrase_variant": row["variant"],
                    "activation_cosine": cosine(
                        prompt_features[ref_index],
                        prompt_features[target_index],
                    ),
                }
            )

    stability_df = pd.DataFrame(rows)
    if stability_df.empty or eval_prompt_summary is None:
        return stability_df

    cols = ["prompt_id", "overall_accuracy", "unseen_mean_accuracy", "transfer_gap"]
    original_metrics = eval_prompt_summary[cols].rename(
        columns={
            "prompt_id": "original_prompt_id",
            "overall_accuracy": "original_overall_accuracy",
            "unseen_mean_accuracy": "original_unseen_accuracy",
            "transfer_gap": "original_transfer_gap",
        }
    )
    paraphrase_metrics = eval_prompt_summary[cols].rename(
        columns={
            "prompt_id": "paraphrase_prompt_id",
            "overall_accuracy": "paraphrase_overall_accuracy",
            "unseen_mean_accuracy": "paraphrase_unseen_accuracy",
            "transfer_gap": "paraphrase_transfer_gap",
        }
    )
    stability_df = stability_df.merge(original_metrics, on="original_prompt_id", how="left")
    stability_df = stability_df.merge(paraphrase_metrics, on="paraphrase_prompt_id", how="left")
    stability_df["unseen_accuracy_delta"] = (
        stability_df["paraphrase_unseen_accuracy"] - stability_df["original_unseen_accuracy"]
    )
    return stability_df


def _make_kfold(n_samples: int, n_splits: int, random_state: int = 42):
    actual_splits = min(n_splits, n_samples)
    if actual_splits < 2:
        return None
    return KFold(n_splits=actual_splits, shuffle=True, random_state=random_state)


def out_of_fold_regression_predictions(X, y, alpha: float = 1.0, n_splits: int = 5, random_state: int = 42):
    splitter = _make_kfold(len(X), n_splits=n_splits, random_state=random_state)
    if splitter is None:
        return np.full(len(y), np.nan), float("nan")

    predictions = np.zeros(len(y), dtype=np.float64)
    for train_index, test_index in splitter.split(X):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                ("ridge", Ridge(alpha=alpha)),
            ]
        )
        model.fit(X[train_index], y[train_index])
        predictions[test_index] = model.predict(X[test_index])

    return predictions, float(r2_score(y, predictions))


def out_of_fold_logistic_accuracy(X, y, c_value: float = 1.0, n_splits: int = 5, random_state: int = 42):
    if len(np.unique(y)) < 2:
        return np.full(len(y), np.nan), float("nan")

    splitter = _make_kfold(len(X), n_splits=n_splits, random_state=random_state)
    if splitter is None:
        return np.full(len(y), np.nan), float("nan")

    predictions = np.zeros(len(y), dtype=np.int64)
    for train_index, test_index in splitter.split(X):
        model = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "logreg",
                    LogisticRegression(
                        C=c_value,
                        max_iter=1000,
                    ),
                ),
            ]
        )
        model.fit(X[train_index], y[train_index])
        predictions[test_index] = model.predict(X[test_index])

    return predictions, float(accuracy_score(y, predictions))


def build_baseline_matrix(eval_prompt_summary, base_prompt_text: str):
    df = eval_prompt_summary.copy()
    df["lexical_similarity_to_base"] = df["prompt_text"].apply(
        lambda text: lexical_similarity(text, base_prompt_text)
    )

    feature_sets = {
        "seen_accuracy": df[["seen_mean_accuracy"]].to_numpy(dtype=np.float64),
        "prompt_meta": df[
            [
                "prompt_length_chars",
                "prompt_length_words",
                "lexical_similarity_to_base",
            ]
        ].to_numpy(dtype=np.float64),
    }

    if "apo_rank" in df.columns:
        feature_sets["apo_rank"] = df[["apo_rank"]].to_numpy(dtype=np.float64)

    return feature_sets, df


def top_k_mean(score, target, k: int):
    if len(score) == 0:
        return float("nan")
    order = np.argsort(score)[::-1]
    top_index = order[: min(k, len(order))]
    return float(np.mean(target[top_index]))


def random_top_k_mean(target, k: int, trials: int = 500, random_state: int = 42):
    if len(target) == 0:
        return float("nan")
    rng = np.random.default_rng(random_state)
    means = []
    for _ in range(trials):
        choice = rng.choice(len(target), size=min(k, len(target)), replace=False)
        means.append(np.mean(target[choice]))
    return float(np.mean(means))


def write_analysis_summary(path, summary: dict):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

