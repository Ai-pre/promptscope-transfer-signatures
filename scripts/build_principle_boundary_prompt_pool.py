from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build third-round boundary-only principle probes."
    )
    parser.add_argument("--output", default="data/prompts_principle_boundary.jsonl")
    return parser.parse_args()


def build_record(
    prompt_id: str,
    text: str,
    *,
    family: str,
    components: list[str],
    complexity_level: str,
    hypothesis_role: str,
    contrast_group: str,
    note: str,
):
    return {
        "id": prompt_id,
        "group_id": prompt_id,
        "variant": "original",
        "source": "principle_probe_boundary",
        "source_title": f"Boundary principle probe: {prompt_id}",
        "source_url": "",
        "paper_title": "",
        "paper_url": "",
        "provenance": "designed_probe",
        "prompt_role": "system",
        "original_prompt_role": "system",
        "task_scope": "task_agnostic",
        "optimized_for_tasks": [],
        "source_datasets": [],
        "source_note": note,
        "principle_family": family,
        "principle_components_json": json.dumps(components, ensure_ascii=False),
        "complexity_level": complexity_level,
        "hypothesis_role": hypothesis_role,
        "contrast_group": contrast_group,
        "text": text,
    }


def build_records():
    return [
        build_record(
            "principle3_plain",
            "Answer the question accurately.",
            family="boundary_only",
            components=[],
            complexity_level="minimal",
            hypothesis_role="baseline",
            contrast_group="plain",
            note="Third-round baseline.",
        ),
        build_record(
            "principle3_concise",
            "Give only the final answer.",
            family="boundary_only",
            components=["concise"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="concise",
            note="Tests whether plain concision alone is sufficient.",
        ),
        build_record(
            "principle3_careful",
            "Read the question carefully and answer accurately.",
            family="boundary_only",
            components=["careful"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="careful",
            note="Tests a lightweight careful-reading boundary.",
        ),
        build_record(
            "principle3_check",
            "Answer the question accurately. Briefly check your answer before finalizing it.",
            family="boundary_only",
            components=["check"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="check",
            note="Tests a lightweight self-check without other cues.",
        ),
        build_record(
            "principle3_concise_format",
            "Give only the final answer. Format it as: FINAL ANSWER: <answer>",
            family="answer_contract",
            components=["concise", "format"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="concise_format",
            note="Strongest performance candidate from round two.",
        ),
        build_record(
            "principle3_careful_format",
            "Read the question carefully and answer accurately. End with: FINAL ANSWER: <answer>",
            family="answer_contract",
            components=["careful", "format"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_format",
            note="Lightweight careful-reading plus answer contract.",
        ),
        build_record(
            "principle3_careful_check",
            "Read the question carefully and answer accurately. Briefly check your answer before finalizing it.",
            family="verification",
            components=["careful", "check"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_check",
            note="Lightweight careful-reading plus brief check.",
        ),
        build_record(
            "principle3_careful_format_check",
            "Read the question carefully and answer accurately. Briefly check your answer before finalizing it. End with: FINAL ANSWER: <answer>",
            family="verification",
            components=["careful", "format", "check"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_format_check",
            note="Strongest activation-alignment candidate from round two.",
        ),
        build_record(
            "principle3_soft_reason",
            "Pause briefly to think before answering, and keep the response focused.",
            family="soft_controls",
            components=["soft_reason"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="soft_reason",
            note="Weak reasoning cue retained because it helped performance in round two.",
        ),
        build_record(
            "principle3_soft_reason_format",
            "Pause briefly to think before answering, keep the response focused, and end with: FINAL ANSWER: <answer>",
            family="soft_controls",
            components=["soft_reason", "format"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="soft_reason_format",
            note="Weak reasoning cue paired with a minimal answer contract.",
        ),
        build_record(
            "principle3_concise_careful_format",
            "Read the question carefully. Give only the final answer, formatted as: FINAL ANSWER: <answer>",
            family="boundary_only",
            components=["concise", "careful", "format"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="concise_careful_format",
            note="Tests whether combining concision with careful-reading and format beats either axis alone.",
        ),
        build_record(
            "principle3_concise_careful_check",
            "Read the question carefully. Give only the final answer. Briefly check it before finalizing it.",
            family="boundary_only",
            components=["concise", "careful", "check"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="concise_careful_check",
            note="Tests whether concision plus careful-reading plus brief checking is enough without an explicit format contract.",
        ),
    ]


def write_jsonl(path, records):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main():
    args = parse_args()
    output_path = resolve_path(".", args.output)
    records = build_records()
    write_jsonl(output_path, records)
    print(f"[build_principle_boundary_prompt_pool] wrote {len(records)} prompts to {output_path}", flush=True)


if __name__ == "__main__":
    main()
