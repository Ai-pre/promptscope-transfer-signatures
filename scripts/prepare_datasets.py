from __future__ import annotations

import argparse
import ast
import json
import random
import re
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

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
        choices=["gsm8k", "csqa", "svamp", "boolq", "bbh", "mmlu_pro", "bbeh_mini", "gpqa"],
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
    parser.add_argument(
        "--hf-token",
        default=None,
        help="Optional Hugging Face token for gated datasets such as Idavidrein/gpqa.",
    )
    return parser.parse_args()


LETTER_LABELS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

BBH_TASKS = [
    "boolean_expressions",
    "causal_judgement",
    "date_understanding",
    "disambiguation_qa",
    "dyck_languages",
    "formal_fallacies",
    "geometric_shapes",
    "hyperbaton",
    "logical_deduction_five_objects",
    "logical_deduction_seven_objects",
    "logical_deduction_three_objects",
    "movie_recommendation",
    "multistep_arithmetic_two",
    "navigate",
    "object_counting",
    "penguins_in_a_table",
    "reasoning_about_colored_objects",
    "ruin_names",
    "salient_translation_error_detection",
    "snarks",
    "sports_understanding",
    "temporal_sequences",
    "tracking_shuffled_objects_five_objects",
    "tracking_shuffled_objects_seven_objects",
    "tracking_shuffled_objects_three_objects",
    "web_of_lies",
    "word_sorting",
]


def source_split(dataset, default="unknown") -> str:
    return str(getattr(dataset, "split", default))


def coerce_options(value):
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        try:
            parsed = ast.literal_eval(stripped)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, (list, tuple)):
            return [str(item) for item in parsed]
        return [stripped]
    return [str(value)]


def format_multiple_choice(question: str, options, *, answer_instruction: str) -> tuple[str, list[str]]:
    choices = coerce_options(options)
    labels = LETTER_LABELS[: len(choices)]
    choice_lines = [f"{label}. {text}" for label, text in zip(labels, choices)]
    return (
        f"Question: {question}\n"
        "Choices:\n"
        + "\n".join(choice_lines)
        + f"\n{answer_instruction}",
        labels,
    )


def interleave_by_key(rows: list[dict], key: str) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(str(row.get(key, "unknown")), []).append(row)

    ordered_keys = sorted(buckets)
    interleaved = []
    while any(buckets.values()):
        for bucket_key in ordered_keys:
            if buckets[bucket_key]:
                interleaved.append(buckets[bucket_key].pop(0))
    return interleaved


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


def convert_gsm8k(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": f"gsm8k-{index}",
                "input": example["question"],
                "label": extract_gsm8k_final_answer(example["answer"]),
                "raw_answer": example["answer"],
                "source_split": source_split(dataset, source_name or "train"),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_csqa(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": example.get("id", f"csqa-{index}"),
                "input": format_csqa_input(example),
                "label": example["answerKey"],
                "valid_choices": list(example["choices"]["label"]),
                "source_split": source_split(dataset, source_name or "validation"),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_svamp(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": example.get("id", f"svamp-{index}"),
                "input": example["question"],
                "label": str(example["result"]),
                "equation": example.get("equation"),
                "problem_type": example.get("problem_type"),
                "source_split": source_split(dataset, source_name or "test"),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_boolq(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": f"boolq-{index}",
                "input": format_boolq_input(example),
                "label": "true" if bool(example["answer"]) else "false",
                "source_split": source_split(dataset, source_name or "validation"),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_bbh(dataset, max_samples=None, source_name=None):
    rows = []
    task_name = source_name or source_split(dataset)
    for index, example in enumerate(dataset):
        rows.append(
            {
                "id": f"bbh-{task_name}-{index}",
                "input": (
                    f"BBH task: {task_name}\n"
                    f"Question: {example['question']}\n"
                    "Answer with the final answer only."
                ),
                "label": str(example["target"]),
                "source_subset": task_name,
                "source_split": source_split(dataset, task_name),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_mmlu_pro(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        question = example["question"]
        options = coerce_options(example["options"])
        formatted, labels = format_multiple_choice(
            question,
            options,
            answer_instruction="Answer with the single best option letter.",
        )
        rows.append(
            {
                "id": f"mmlu-pro-{example.get('question_id', index)}",
                "input": formatted,
                "label": str(example["answer"]).strip(),
                "valid_choices": labels,
                "category": example.get("category"),
                "source_subset": example.get("category", "mmlu_pro"),
                "source_split": source_split(dataset, source_name or "test"),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return rows


def convert_bbeh_mini(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        try:
            is_mini = int(example.get("mini", 0)) == 1
        except (TypeError, ValueError):
            is_mini = str(example.get("mini", "")).strip().lower() in {"1", "true", "yes"}
        if not is_mini:
            continue

        question = example.get("input") or example.get("question") or example.get("prompt")
        label = example.get("target") or example.get("answer") or example.get("label")
        subset = example.get("task") or example.get("task_name") or "bbeh_mini"
        rows.append(
            {
                "id": f"bbeh-mini-{subset}-{index}",
                "input": (
                    f"BBEH task: {subset}\n"
                    f"Question: {question}\n"
                    "Answer with the final answer only."
                ),
                "label": str(label),
                "source_subset": str(subset),
                "source_split": source_split(dataset, source_name or "train"),
            }
        )
        if max_samples is not None and len(rows) >= max_samples:
            break
    return interleave_by_key(rows, "source_subset")


def convert_gpqa(dataset, max_samples=None, source_name=None):
    rows = []
    for index, example in enumerate(dataset):
        if "Correct Answer" in example:
            choices = [
                example["Correct Answer"],
                example["Incorrect Answer 1"],
                example["Incorrect Answer 2"],
                example["Incorrect Answer 3"],
            ]
            shuffled = list(enumerate(choices))
            random.Random(index).shuffle(shuffled)
            options = [choice for _, choice in shuffled]
            correct_position = next(pos for pos, (original_idx, _) in enumerate(shuffled) if original_idx == 0)
            label = LETTER_LABELS[correct_position]
            question = example["Question"]
            source_subset = example.get("High-level domain") or example.get("Subdomain") or "gpqa"
        else:
            question = example["question"]
            solution = str(example.get("solution", "")).strip()
            answer_match = re.search(r"Answer:\s*([A-D])", solution, flags=re.IGNORECASE)
            label = answer_match.group(1).upper() if answer_match else solution
            options = []
            source_subset = example.get("source_type", "gpqa_diamond")

        if options:
            formatted, labels = format_multiple_choice(
                question,
                options,
                answer_instruction="Answer with the single best option letter.",
            )
        else:
            formatted = f"Question: {question}\nAnswer with the single best option letter."
            labels = ["A", "B", "C", "D"]

        rows.append(
            {
                "id": f"gpqa-{index}",
                "input": formatted,
                "label": label,
                "valid_choices": labels,
                "source_subset": str(source_subset),
                "source_split": source_split(dataset, source_name or "train"),
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
    "bbh": {
        "path": "Joschka/big_bench_hard",
        "configs": BBH_TASKS,
        "split": None,
        "converter": convert_bbh,
        "notes": "Combines the 27 non-prompt BIG-Bench Hard subsets as a broad in-domain reasoning pool.",
    },
    "mmlu_pro": {
        "path": "TIGER-Lab/MMLU-Pro",
        "name": None,
        "split": "test",
        "converter": convert_mmlu_pro,
        "notes": "Uses MMLU-Pro test as a broad knowledge-intensive seen-task pool.",
    },
    "bbeh_mini": {
        "path": "BBEH/bbeh",
        "name": None,
        "split": "train",
        "converter": convert_bbeh_mini,
        "notes": "Uses the mini subset of BIG-Bench Extra Hard as a harder OOD pair for BBH.",
    },
    "gpqa": {
        "path": "Idavidrein/gpqa",
        "name": "gpqa_diamond",
        "split": "train",
        "converter": convert_gpqa,
        "fallbacks": [
            {
                "path": "nichenshun/gpqa_diamond",
                "name": None,
                "split": "train",
                "converter": convert_gpqa,
            }
        ],
        "notes": "Uses GPQA Diamond as a difficult OOD science reasoning benchmark. Falls back to a public GPQA Diamond mirror if the official gated repo is unavailable.",
    },
}


def pick_dataset_split(dataset_dict, split_name: str | None, fallback_name: str | None = None):
    if not hasattr(dataset_dict, "keys"):
        return dataset_dict
    if split_name and split_name in dataset_dict:
        return dataset_dict[split_name]
    if fallback_name and fallback_name in dataset_dict:
        return dataset_dict[fallback_name]
    first_key = next(iter(dataset_dict.keys()))
    return dataset_dict[first_key]


def load_remote_dataset_from_spec(source_spec: dict, cache_dir: str | None = None, hf_token: str | None = None, fallback_split=None):
    from datasets import load_dataset

    split_name = source_spec.get("split")
    kwargs = {}
    if split_name is not None:
        kwargs["split"] = split_name
    if source_spec.get("name") is not None:
        kwargs["name"] = source_spec["name"]
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if hf_token is not None:
        kwargs["token"] = hf_token
    try:
        return load_dataset(source_spec["path"], **kwargs)
    except ValueError:
        kwargs.pop("split", None)
        dataset_dict = load_dataset(source_spec["path"], **kwargs)
        return pick_dataset_split(dataset_dict, split_name, fallback_name=fallback_split)


def load_remote_dataset(task_name: str, cache_dir: str | None = None, hf_token: str | None = None, source_name: str | None = None):
    spec = DATASET_SPECS[task_name]
    source_spec = dict(spec)
    if source_name is not None:
        source_spec["name"] = source_name
        source_spec["split"] = spec.get("split") or source_name
    return load_remote_dataset_from_spec(source_spec, cache_dir=cache_dir, hf_token=hf_token, fallback_split=source_name)


def load_single_source_with_fallbacks(task_name: str, cache_dir: str | None = None, hf_token: str | None = None):
    spec = DATASET_SPECS[task_name]
    candidates = [spec] + list(spec.get("fallbacks", []))
    errors = []

    for source_spec in candidates:
        try:
            dataset = load_remote_dataset_from_spec(
                source_spec,
                cache_dir=cache_dir,
                hf_token=hf_token,
                fallback_split=source_spec.get("name"),
            )
            return dataset, source_spec
        except Exception as exc:  # noqa: BLE001 - fallback needs to catch gated/HTTP/schema errors.
            errors.append(f"{source_spec['path']}: {type(exc).__name__}: {exc}")
            print(
                f"[prepare_datasets] warning: failed to load {task_name} from {source_spec['path']}: {exc}",
                flush=True,
            )

    raise RuntimeError(f"Could not load {task_name} from any configured source: {' | '.join(errors)}")


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
        rows = []
        used_sources = []

        if spec.get("configs"):
            for source_name in spec["configs"]:
                print(
                    f"[prepare_datasets] downloading {task_name}/{source_name} from {spec['path']}",
                    flush=True,
                )
                try:
                    remote_dataset = load_remote_dataset(
                        task_name,
                        cache_dir=args.cache_dir,
                        hf_token=args.hf_token,
                        source_name=source_name,
                    )
                except Exception as exc:  # noqa: BLE001 - keep the aggregate benchmark usable if one subset name drifts.
                    print(
                        f"[prepare_datasets] warning: skipping {task_name}/{source_name}: {exc}",
                        flush=True,
                    )
                    continue
                rows.extend(spec["converter"](remote_dataset, max_samples=None, source_name=source_name))
                used_sources.append({"path": spec["path"], "name": source_name, "split": spec.get("split") or source_name})
            rows = interleave_by_key(rows, "source_subset")
            if args.max_samples is not None:
                rows = rows[: args.max_samples]
            if not rows:
                raise RuntimeError(f"No rows were loaded for aggregate dataset {task_name}.")
        else:
            print(
                f"[prepare_datasets] downloading {task_name} from {spec['path']} split={spec.get('split')}",
                flush=True,
            )
            remote_dataset, used_spec = load_single_source_with_fallbacks(
                task_name,
                cache_dir=args.cache_dir,
                hf_token=args.hf_token,
            )
            converter = used_spec.get("converter", spec["converter"])
            rows = converter(
                remote_dataset,
                max_samples=args.max_samples,
                source_name=used_spec.get("name") or used_spec.get("split"),
            )
            used_sources.append(
                {
                    "path": used_spec["path"],
                    "name": used_spec.get("name"),
                    "split": used_spec.get("split"),
                }
            )

        output_path = output_dir / f"{task_name}.json"
        write_json_dataset(output_path, rows)
        manifest[task_name] = {
            "path": spec["path"],
            "name": spec.get("name"),
            "split": spec.get("split"),
            "used_sources": used_sources,
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
