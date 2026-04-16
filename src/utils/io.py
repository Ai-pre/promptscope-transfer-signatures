from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_PROMPT_FIELDS = {"id", "text", "source"}
REQUIRED_SAMPLE_FIELDS = {"input", "label"}
OPTIONAL_PROMPT_LIST_FIELDS = ("optimized_for_tasks", "source_datasets")


def load_config(path):
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["_config_path"] = str(config_path)
    config["_project_root"] = str(config_path.parent.parent)
    return config


def resolve_path(project_root, relative_or_absolute):
    path = Path(relative_or_absolute)
    if path.is_absolute():
        return path
    return Path(project_root) / path


def ensure_dir(path):
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def save_json(path, payload):
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


def save_markdown(path, text: str):
    with Path(path).open("w", encoding="utf-8") as handle:
        handle.write(text)


def save_dataframe(df: pd.DataFrame, path):
    df.to_parquet(path, index=False)


def load_prompts(path):
    prompts = load_jsonl(path)
    normalized = []
    for prompt in prompts:
        missing = REQUIRED_PROMPT_FIELDS - set(prompt)
        if missing:
            raise ValueError(f"Prompt record missing required fields {sorted(missing)}: {prompt}")
        record = dict(prompt)
        record.setdefault("group_id", record["id"])
        record.setdefault("variant", "original")
        for field_name in OPTIONAL_PROMPT_LIST_FIELDS:
            record[field_name] = normalize_prompt_list_field(record.get(field_name))
            record[f"{field_name}_json"] = json.dumps(record[field_name], ensure_ascii=False)
        record.setdefault("prompt_role", "system")
        record.setdefault("original_prompt_role", record["prompt_role"])
        record.setdefault("task_scope", "task_specific" if record["optimized_for_tasks"] else "task_agnostic")
        record.setdefault("provenance", "legacy_starter")
        record.setdefault("source_title", "")
        record.setdefault("source_url", "")
        record.setdefault("paper_title", "")
        record.setdefault("paper_url", "")
        record.setdefault("source_note", "")
        record["is_paper_backed"] = bool(record["source_url"] or record["paper_url"])
        record["prompt_length_chars"] = len(record["text"])
        record["prompt_length_words"] = len(record["text"].split())
        normalized.append(record)
    return normalized


def normalize_prompt_list_field(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, list):
                return [str(item).strip() for item in parsed if str(item).strip()]
        return [piece.strip() for piece in re.split(r"[;,]", stripped) if piece.strip()]
    return [str(value).strip()]


def ensure_base_prompt_record(prompts, base_prompt: str):
    if any(prompt["text"] == base_prompt for prompt in prompts):
        return prompts
    base_record = {
        "id": "base_prompt",
        "group_id": "base_prompt",
        "variant": "original",
        "source": "base",
        "source_title": "Neutral base prompt",
        "source_url": "",
        "paper_title": "",
        "paper_url": "",
        "provenance": "base",
        "prompt_role": "system",
        "original_prompt_role": "system",
        "task_scope": "task_agnostic",
        "optimized_for_tasks": [],
        "optimized_for_tasks_json": "[]",
        "source_datasets": [],
        "source_datasets_json": "[]",
        "source_note": "Neutral reference prompt used to compute delta_h.",
        "is_paper_backed": False,
        "text": base_prompt,
        "prompt_length_chars": len(base_prompt),
        "prompt_length_words": len(base_prompt.split()),
    }
    return prompts + [base_record]


def load_dataset(path, task_name: str, limit=None):
    path = Path(path)
    if path.suffix == ".jsonl":
        records = load_jsonl(path)
    else:
        records = load_json(path)

    if not isinstance(records, list):
        raise ValueError(f"Dataset at {path} must be a list of samples.")

    normalized = []
    for index, sample in enumerate(records):
        missing = REQUIRED_SAMPLE_FIELDS - set(sample)
        if missing:
            raise ValueError(f"Dataset sample missing required fields {sorted(missing)}: {sample}")
        record = dict(sample)
        record.setdefault("id", f"{task_name}-{index}")
        record["task"] = task_name
        normalized.append(record)

    if limit is not None:
        normalized = normalized[: int(limit)]
    return normalized


def flatten_task_config(config):
    seen = list(config["tasks"]["seen"])
    unseen = list(config["tasks"]["unseen"])
    ordered = []
    for task in seen + unseen:
        if task not in ordered:
            ordered.append(task)
    return ordered


def batched(items, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    for index in range(0, len(items), batch_size):
        yield items[index : index + batch_size]


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def vector_rows_to_table(rows, vector_key: str):
    if not rows:
        return np.empty((0, 0), dtype=np.float32), pd.DataFrame()

    metadata = []
    vectors = []
    for row in rows:
        copied = dict(row)
        vectors.append(np.asarray(copied.pop(vector_key), dtype=np.float32))
        metadata.append(copied)
    return np.stack(vectors), pd.DataFrame(metadata)
