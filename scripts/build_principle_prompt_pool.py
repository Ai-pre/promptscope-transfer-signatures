from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build minimal principle-probe prompts for transfer/activation hypothesis testing."
    )
    parser.add_argument("--output", default="data/prompts_principle_minimal.jsonl")
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
        "source": "principle_probe",
        "source_title": f"Principle probe: {prompt_id}",
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


def build_principle_records():
    return [
        build_record(
            "principle_plain",
            "Answer the question accurately.",
            family="factorial",
            components=[],
            complexity_level="minimal",
            hypothesis_role="baseline",
            contrast_group="plain",
            note="Minimal baseline with no explicit expert, reasoning, format, or verification instruction.",
        ),
        build_record(
            "principle_expert_only",
            "You are a knowledgeable expert. Answer the question accurately.",
            family="factorial",
            components=["expert"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="expert",
            note="Tests whether a generic expert stance alone improves transfer and activation alignment.",
        ),
        build_record(
            "principle_reasoning_only",
            "Think through the problem step by step before answering.",
            family="factorial",
            components=["reasoning"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="reasoning",
            note="Tests whether an explicit reasoning mode alone improves transfer and activation alignment.",
        ),
        build_record(
            "principle_format_only",
            "Answer the question accurately. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="factorial",
            components=["format"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="format",
            note="Tests whether a lightweight final-answer contract alone improves transfer and activation alignment.",
        ),
        build_record(
            "principle_expert_reasoning",
            "You are a knowledgeable expert. Think through the problem step by step before answering.",
            family="factorial",
            components=["expert", "reasoning"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="expert_reasoning",
            note="Tests whether combining expert stance and reasoning is stronger than either alone.",
        ),
        build_record(
            "principle_expert_format",
            "You are a knowledgeable expert. Answer the question accurately. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="factorial",
            components=["expert", "format"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="expert_format",
            note="Tests whether combining expert stance with a final-answer contract is stronger than either alone.",
        ),
        build_record(
            "principle_reasoning_format",
            "Think through the problem step by step before answering. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="factorial",
            components=["reasoning", "format"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="reasoning_format",
            note="Tests whether combining reasoning mode with a final-answer contract is stronger than either alone.",
        ),
        build_record(
            "principle_expert_reasoning_format",
            "You are a knowledgeable expert. Think through the problem step by step before answering. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="factorial",
            components=["expert", "reasoning", "format"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="expert_reasoning_format",
            note="Main principle probe: lightweight expert stance plus reasoning mode plus answer contract.",
        ),
        build_record(
            "principle_expert_reasoning_format_verify",
            "You are a knowledgeable expert. Think through the problem step by step before answering. Briefly verify your answer before finalizing it. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="verification",
            components=["expert", "reasoning", "format", "verify"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="expert_reasoning_format_verify",
            note="Tests whether adding a short verification step further improves the lightweight principle prompt.",
        ),
        build_record(
            "principle_answer_then_check",
            "Answer the question accurately. Briefly check your answer before finalizing it. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="verification",
            components=["format", "verify"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="format_verify",
            note="Tests whether a lightweight verify-and-format contract helps without explicit expert or reasoning instructions.",
        ),
        build_record(
            "principle_verbose_expert_reasoning_format",
            "You are a world-class interdisciplinary expert with broad knowledge across mathematics, commonsense reasoning, and language understanding. Carefully analyze the task, consider the relevant facts, reason step by step, and maintain a high standard of rigor throughout your solution. Organize your thinking clearly, avoid unnecessary assumptions, and be precise in your final response. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="scaffold_controls",
            components=["expert", "reasoning", "format", "verbose"],
            complexity_level="heavy",
            hypothesis_role="control",
            contrast_group="verbose_control",
            note="Heavy control with the same core ingredients as the lightweight hypothesis prompt but extra world-building and verbosity.",
        ),
        build_record(
            "principle_multiagent_scaffold",
            "You are a coordinator working with a Solver, a Critic, and a Verifier. First assign roles, then let the Solver reason step by step, let the Critic challenge the reasoning, and let the Verifier check the answer before you respond. End with a line formatted exactly as: FINAL ANSWER: <answer>",
            family="scaffold_controls",
            components=["reasoning", "format", "verify", "multiagent"],
            complexity_level="heavy",
            hypothesis_role="control",
            contrast_group="multiagent_control",
            note="Heavy multi-agent scaffold control that tests whether explicit orchestration hurts transfer relative to lightweight boundary setting.",
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
    records = build_principle_records()
    write_jsonl(output_path, records)
    print(f"[build_principle_prompt_pool] wrote {len(records)} prompts to {output_path}", flush=True)


if __name__ == "__main__":
    main()
