from __future__ import annotations

import copy
import re
from decimal import Decimal, InvalidOperation

import pandas as pd
import torch

from src.model.load_model import get_model_device
from src.prompt.prompt_builder import build_input
from src.utils.io import batched


NUMERIC_TASKS = {"gsm8k", "svamp"}
MULTIPLE_CHOICE_TASKS = {"csqa", "mmlu_pro", "gpqa"}
BOOLEAN_TASKS = {"boolq"}
SHORT_ANSWER_TASKS = {"bbh", "bbeh_mini"}


def generate_batch(model, tokenizer, prompts, max_new_tokens: int = 128):
    device = get_model_device(model)
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        add_special_tokens=False,
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    prompt_length = encoded["input_ids"].shape[1]

    generation_config = getattr(model, "generation_config", None)
    generation_kwargs = {}
    if generation_config is not None:
        # Keep generation deterministic and warning-free across Transformers
        # versions by passing all generation settings through one config object.
        generation_config = copy.deepcopy(generation_config)
        generation_config.max_length = prompt_length + max_new_tokens
        generation_config.max_new_tokens = None
        generation_config.do_sample = False
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config.eos_token_id = tokenizer.eos_token_id
        for sampling_field in ("temperature", "top_p", "top_k"):
            if hasattr(generation_config, sampling_field):
                setattr(generation_config, sampling_field, None)
        generation_kwargs["generation_config"] = generation_config
    else:
        generation_kwargs.update(
            {
                "max_length": prompt_length + max_new_tokens,
                "do_sample": False,
                "pad_token_id": tokenizer.pad_token_id,
                "eos_token_id": tokenizer.eos_token_id,
            }
        )

    with torch.no_grad():
        output_ids = model.generate(
            **encoded,
            **generation_kwargs,
        )

    generations = []
    for row in output_ids:
        generated_tokens = row[prompt_length:]
        generations.append(tokenizer.decode(generated_tokens, skip_special_tokens=True).strip())
    return generations


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128):
    return generate_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt],
        max_new_tokens=max_new_tokens,
    )[0]


def normalize_text(text) -> str:
    return re.sub(r"\s+", " ", str(text).strip().lower())


def canonicalize_number(text) -> str:
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", str(text))
    if not matches:
        return normalize_text(text)
    value = matches[-1].replace(",", "")
    try:
        return str(Decimal(value).normalize())
    except InvalidOperation:
        return value


def canonicalize_choice(text, valid_choices=None) -> str:
    normalized = str(text).upper()
    choices = valid_choices or ["A", "B", "C", "D", "E"]
    pattern = r"\b(" + "|".join(re.escape(choice) for choice in choices) + r")\b"
    matches = re.findall(pattern, normalized)
    if matches:
        return matches[-1]
    return normalize_text(text)


def canonicalize_boolean(text) -> str:
    normalized = normalize_text(text)
    if normalized in {"true", "yes", "1"}:
        return "true"
    if normalized in {"false", "no", "0"}:
        return "false"

    matches = re.findall(r"\b(true|false|yes|no)\b", normalized)
    if not matches:
        return normalized
    return "true" if matches[-1] in {"true", "yes"} else "false"


def canonicalize_short_answer(text) -> str:
    normalized = normalize_text(text)
    normalized = re.sub(
        r"^(final\s+answer|answer|the\s+answer\s+is|therefore,?\s+the\s+answer\s+is)\s*[:\-]?\s*",
        "",
        normalized,
    )
    normalized = normalized.splitlines()[-1] if "\n" in normalized else normalized
    return normalized.strip(" .,:;`'\"()[]{}")


def short_answer_matches(prediction: str, gold_label) -> tuple[float, str, str]:
    pred_norm = canonicalize_short_answer(prediction)
    gold_norm = canonicalize_short_answer(gold_label)
    if pred_norm == gold_norm:
        return 1.0, pred_norm, gold_norm

    # Avoid false positives for single-letter answers; those should be handled
    # by the multiple-choice normalizer when valid choices are available.
    if len(gold_norm) <= 1:
        return 0.0, pred_norm, gold_norm

    pattern = r"(?<!\w)" + re.escape(gold_norm) + r"(?!\w)"
    return float(bool(re.search(pattern, normalize_text(prediction)))), pred_norm, gold_norm


def normalize_prediction(task_name: str, prediction: str, gold_label, sample: dict):
    if task_name in NUMERIC_TASKS:
        return canonicalize_number(prediction), canonicalize_number(gold_label)
    if task_name in MULTIPLE_CHOICE_TASKS:
        valid_choices = sample.get("valid_choices")
        return (
            canonicalize_choice(prediction, valid_choices=valid_choices),
            canonicalize_choice(gold_label, valid_choices=valid_choices),
        )
    if task_name in BOOLEAN_TASKS:
        return canonicalize_boolean(prediction), canonicalize_boolean(gold_label)
    if task_name in SHORT_ANSWER_TASKS:
        return canonicalize_short_answer(prediction), canonicalize_short_answer(gold_label)
    return normalize_text(prediction), normalize_text(gold_label)


def compute_accuracy(prediction: str, gold_label, task_name: str, sample: dict):
    if task_name in SHORT_ANSWER_TASKS:
        return short_answer_matches(prediction, gold_label)

    pred_norm, gold_norm = normalize_prediction(task_name, prediction, gold_label, sample)
    return float(pred_norm == gold_norm), pred_norm, gold_norm


def evaluate_dataset(
    model,
    tokenizer,
    dataset,
    system_prompt: str,
    prompt_record: dict,
    task_name: str,
    split: str,
    batch_size: int = 8,
    max_new_tokens: int = 128,
):
    rows = []

    for batch in batched(dataset, batch_size):
        prompts = [build_input(tokenizer, system_prompt, sample["input"]) for sample in batch]
        predictions = generate_batch(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            max_new_tokens=max_new_tokens,
        )

        for sample, prediction in zip(batch, predictions):
            correct, pred_norm, gold_norm = compute_accuracy(
                prediction=prediction,
                gold_label=sample["label"],
                task_name=task_name,
                sample=sample,
            )
            rows.append(
                {
                    "prompt_id": prompt_record["id"],
                    "group_id": prompt_record.get("group_id", prompt_record["id"]),
                    "variant": prompt_record.get("variant", "original"),
                    "source": prompt_record["source"],
                    "prompt_text": prompt_record["text"],
                    "prompt_length_chars": prompt_record.get("prompt_length_chars", len(prompt_record["text"])),
                    "prompt_length_words": prompt_record.get("prompt_length_words", len(prompt_record["text"].split())),
                    "task": task_name,
                    "split": split,
                    "sample_id": sample["id"],
                    "input": sample["input"],
                    "label": str(sample["label"]),
                    "prediction": prediction,
                    "normalized_prediction": pred_norm,
                    "normalized_label": gold_norm,
                    "correct": correct,
                }
            )

    return rows


def summarize_prompt_metrics(sample_rows, seen_tasks, unseen_tasks):
    sample_df = pd.DataFrame(sample_rows)
    if sample_df.empty:
        return sample_df, sample_df

    prompt_columns = [
        "prompt_id",
        "group_id",
        "variant",
        "source",
        "prompt_text",
        "prompt_length_chars",
        "prompt_length_words",
    ]

    task_summary = (
        sample_df.groupby(
            prompt_columns + ["task", "split"],
            dropna=False,
        )["correct"]
        .mean()
        .reset_index(name="accuracy")
    )

    prompt_rows = []
    for prompt_key, group in task_summary.groupby(prompt_columns, dropna=False):
        record = {column: value for column, value in zip(prompt_columns, prompt_key)}
        seen_acc = group.loc[group["task"].isin(seen_tasks), "accuracy"]
        unseen_acc = group.loc[group["task"].isin(unseen_tasks), "accuracy"]
        record["seen_mean_accuracy"] = float(seen_acc.mean()) if not seen_acc.empty else float("nan")
        record["unseen_mean_accuracy"] = float(unseen_acc.mean()) if not unseen_acc.empty else float("nan")
        record["overall_accuracy"] = float(group["accuracy"].mean())
        record["transfer_gap"] = record["seen_mean_accuracy"] - record["unseen_mean_accuracy"]
        prompt_rows.append(record)

    prompt_summary = pd.DataFrame(prompt_rows)
    sensitivity = (
        task_summary.groupby(["group_id", "task"], dropna=False)["accuracy"]
        .var()
        .fillna(0.0)
        .reset_index(name="task_paraphrase_variance")
    )
    group_sensitivity = (
        sensitivity.groupby("group_id", dropna=False)["task_paraphrase_variance"]
        .mean()
        .reset_index(name="sensitivity")
    )
    prompt_summary = prompt_summary.merge(group_sensitivity, on="group_id", how="left")
    prompt_summary["sensitivity"] = prompt_summary["sensitivity"].fillna(0.0)

    return task_summary, prompt_summary
