from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, load_config, load_prompts, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build filtered prompt pools for strict and mixed transfer experiments."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None, help="Optional prompt file override.")
    parser.add_argument("--output-dir", default="data")
    parser.add_argument(
        "--tag",
        default="",
        help="Optional suffix inserted before .jsonl so multiple seen-task prompt pools can coexist.",
    )
    return parser.parse_args()


def write_jsonl(path, records):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def tagged_name(file_name: str, tag: str) -> str:
    if not tag:
        return file_name
    if file_name.endswith(".jsonl"):
        return f"{file_name[:-6]}_{tag}.jsonl"
    return f"{file_name}_{tag}"


def intersects_seen_tasks(record: dict, seen_tasks: set[str]) -> bool:
    optimized = set(record.get("optimized_for_tasks", []))
    return bool(seen_tasks & optimized)


def has_suspicious_text_artifacts(text: str) -> bool:
    if "\ufffd" in text:
        return True
    for char in text:
        code = ord(char)
        if 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F or 0xAC00 <= code <= 0xD7AF:
            return True
    return False


def is_clean_prompt(record: dict) -> bool:
    return not has_suspicious_text_artifacts(record.get("text", ""))


def filter_records(records, *, predicate):
    return [record for record in records if predicate(record)]


def main():
    args = parse_args()
    config = load_config(args.config)
    project_root = config["_project_root"]
    prompt_path = resolve_path(
        project_root,
        args.input if args.input is not None else config["paths"]["prompts"],
    )
    output_dir = ensure_dir(resolve_path(project_root, args.output_dir))

    records = [record for record in load_prompts(prompt_path) if record["source"] != "base"]
    seen_tasks = set(config["tasks"]["seen"])

    subsets = {
        "prompts_paper_backed_original_only.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original" and bool(record.get("is_paper_backed", False)),
        ),
        "prompts_paper_backed_mixed_seen_aligned.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and (record.get("task_scope") == "task_agnostic" or intersects_seen_tasks(record, seen_tasks)),
        ),
        "prompts_paper_backed_strict_system_only.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and record.get("original_prompt_role") == "system",
        ),
        "prompts_paper_backed_strict_system_seen_aligned.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and record.get("original_prompt_role") == "system"
            and (record.get("task_scope") == "task_agnostic" or intersects_seen_tasks(record, seen_tasks)),
        ),
        "prompts_paper_backed_original_only_clean.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and is_clean_prompt(record),
        ),
        "prompts_paper_backed_mixed_seen_aligned_clean.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and is_clean_prompt(record)
            and (record.get("task_scope") == "task_agnostic" or intersects_seen_tasks(record, seen_tasks)),
        ),
        "prompts_paper_backed_strict_system_only_clean.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and record.get("original_prompt_role") == "system"
            and is_clean_prompt(record),
        ),
        "prompts_paper_backed_strict_system_seen_aligned_clean.jsonl": filter_records(
            records,
            predicate=lambda record: record.get("variant", "original") == "original"
            and bool(record.get("is_paper_backed", False))
            and record.get("original_prompt_role") == "system"
            and is_clean_prompt(record)
            and (record.get("task_scope") == "task_agnostic" or intersects_seen_tasks(record, seen_tasks)),
        ),
    }

    for file_name, subset in subsets.items():
        output_path = output_dir / tagged_name(file_name, args.tag)
        write_jsonl(output_path, subset)
        print(f"[build_prompt_subsets] wrote {len(subset)} prompts to {output_path}", flush=True)


if __name__ == "__main__":
    main()
