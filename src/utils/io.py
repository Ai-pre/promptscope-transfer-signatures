from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


REQUIRED_PROMPT_FIELDS = {"id", "text", "source"}
REQUIRED_SAMPLE_FIELDS = {"input", "label"}


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
        record["prompt_length_chars"] = len(record["text"])
        record["prompt_length_words"] = len(record["text"].split())
        normalized.append(record)
    return normalized


def ensure_base_prompt_record(prompts, base_prompt: str):
    if any(prompt["text"] == base_prompt for prompt in prompts):
        return prompts
    base_record = {
        "id": "base_prompt",
        "group_id": "base_prompt",
        "variant": "original",
        "source": "base",
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

