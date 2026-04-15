from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from datasets import load_dataset

from src.utils.io import ensure_dir, load_config, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download canonical benchmark datasets from Hugging Face and convert them into the local JSON schema."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--tasks",
        nargs="*",
        default=["gsm8k", "csqa", "svamp", "boolq"],
        choices=["gsm8k", "csqa", "svamp", "boolq"],
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap per dataset after conversion.",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face datasets cache dir.",
    )
    return parser.parse_args()


def extract_gsm8k_final_answer(answer_text: str) -> str:
    match = re.search(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", answer_text)
    if match:
        return match.group(1).replace(",", "")

    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", answer_text)
    if not matches:
        raise ValueError(f"Could not parse GSM8K answer: {answer_text!r}")
    return matches[-1].replace(",", "")


def format_csqa_input(example: dict) -> str:
    labels = example["choices"]["label"]
    texts = example["choices"]["text"]
    choice_lines = [f"{label}. {text}" for label, text in zip(labels, texts)]
    return (
        f"Question: {example['question']}\n"
        "Choices:\n"
        + "\n".join(choice_lines)
        + "\nAnswer with the single best option."
    )


def format_boolq_input(example: dict) -> str:
    return (
        f"Passage: {example['passage']}\n"
        f"Question: {example['question']}\n"
        "Answer true or false."
    )


def convert_gsm8k(dataset, max_samples=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": f"gsm8k-{index}",
                "input": example["question"],
                "label": extract_gsm8k_final_answer(example["answer"]),
                "raw_answer": example["answer"],
                "source_split": str(dataset.split),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_csqa(dataset, max_samples=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": example.get("id", f"csqa-{index}"),
                "input": format_csqa_input(example),
                "label": example["answerKey"],
                "valid_choices": list(example["choices"]["label"]),
                "source_split": str(dataset.split),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_svamp(dataset, max_samples=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": example.get("id", f"svamp-{index}"),
                "input": example["question"],
                "label": str(example["result"]),
                "equation": example.get("equation"),
                "problem_type": example.get("problem_type"),
                "source_split": str(dataset.split),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_boolq(dataset, max_samples=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": f"boolq-{index}",
                "input": format_boolq_input(example),
                "label": "true" if bool(example["answer"]) else "false",
                "source_split": str(dataset.split),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


DATASET_SPECS = {
    "gsm8k": {
        "path": "openai/gsm8k",
        "name": "main",
        "split": "train",
        "converter": convert_gsm8k,
        "notes": "Uses the labeled training split as a prompt-selection pool for seen-task evaluation.",
    },
    "csqa": {
        "path": "tau/commonsense_qa",
        "name": None,
        "split": "validation",
        "converter": convert_csqa,
        "notes": "Uses the labeled validation split because the benchmark test split is unlabeled.",
    },
    "svamp": {
        "path": "MU-NLPC/Calc-svamp",
        "name": None,
        "split": "test",
        "converter": convert_svamp,
        "notes": "Uses the labeled test split from the Calc-SVAMP variant on Hugging Face.",
    },
    "boolq": {
        "path": "google/boolq",
        "name": None,
        "split": "validation",
        "converter": convert_boolq,
        "notes": "Uses the labeled validation split because the benchmark test split is unlabeled.",
    },
}


def load_remote_dataset(task_name: str, cache_dir: str | None = None):
    spec = DATASET_SPECS[task_name]
    kwargs = {"split": spec["split"]}
    if spec["name"] is not None:
        kwargs["name"] = spec["name"]
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    return load_dataset(spec["path"], **kwargs)


def write_json_dataset(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    output_dir = ensure_dir(resolve_path(project_root, config["paths"]["datasets_dir"]))

    manifest = {}
    for task_name in args.tasks:
        spec = DATASET_SPECS[task_name]
        print(
            f"[prepare_datasets] downloading {task_name} from {spec['path']} split={spec['split']}",
            flush=True,
        )
        remote_dataset = load_remote_dataset(task_name, cache_dir=args.cache_dir)
        rows = spec["converter"](remote_dataset, max_samples=args.max_samples)
        output_path = output_dir / f"{task_name}.json"
        write_json_dataset(output_path, rows)
        manifest[task_name] = {
            "path": spec["path"],
            "name": spec["name"],
            "split": spec["split"],
            "num_rows": len(rows),
            "output_file": str(output_path),
            "notes": spec["notes"],
        }
        print(
            f"[prepare_datasets] wrote {len(rows)} rows to {output_path}",
            flush=True,
        )

    manifest_path = output_dir / "manifest.json"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(f"[prepare_datasets] wrote manifest to {manifest_path}", flush=True)


if __name__ == "__main__":
    main()

