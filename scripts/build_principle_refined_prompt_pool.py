from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build refined second-round principle probes focused on lightweight boundary-setting."
    )
    parser.add_argument("--output", default="data/prompts_principle_refined.jsonl")
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
        "source": "principle_probe_refined",
        "source_title": f"Refined principle probe: {prompt_id}",
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
            "principle2_plain",
            "Answer the question accurately.",
            family="boundary_setting",
            components=[],
            complexity_level="minimal",
            hypothesis_role="baseline",
            contrast_group="plain",
            note="Minimal baseline for the refined second-round principle study.",
        ),
        build_record(
            "principle2_careful",
            "Read the question carefully and answer accurately.",
            family="boundary_setting",
            components=["careful"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="careful",
            note="Tests whether a lightweight careful-reading boundary helps without heavy scaffolding.",
        ),
        build_record(
            "principle2_format_only",
            "Answer the question accurately. End with: FINAL ANSWER: <answer>",
            family="answer_contract",
            components=["format"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="format",
            note="Tests a minimal final-answer contract without other guidance.",
        ),
        build_record(
            "principle2_careful_format",
            "Read the question carefully and answer accurately. End with: FINAL ANSWER: <answer>",
            family="answer_contract",
            components=["careful", "format"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_format",
            note="Main refined hypothesis: lightweight careful-reading plus answer contract.",
        ),
        build_record(
            "principle2_check_only",
            "Answer the question accurately. Briefly check your answer before finalizing it.",
            family="verification",
            components=["verify"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="check",
            note="Tests whether a lightweight verification step helps by itself.",
        ),
        build_record(
            "principle2_careful_check",
            "Read the question carefully and answer accurately. Briefly check your answer before finalizing it.",
            family="verification",
            components=["careful", "verify"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_check",
            note="Tests whether careful-reading plus a brief check improves the lightweight principle.",
        ),
        build_record(
            "principle2_careful_format_check",
            "Read the question carefully and answer accurately. Briefly check your answer before finalizing it. End with: FINAL ANSWER: <answer>",
            family="verification",
            components=["careful", "format", "verify"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_format_check",
            note="Tests the full lightweight boundary-setting hypothesis with careful reading, answer contract, and brief verification.",
        ),
        build_record(
            "principle2_concise_format",
            "Give only the final answer. Format it as: FINAL ANSWER: <answer>",
            family="boundary_setting",
            components=["concise", "format"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="concise_format",
            note="Tests whether minimality plus a simple answer contract is sufficient.",
        ),
        build_record(
            "principle2_soft_reason",
            "Pause briefly to think before answering, and keep the response focused.",
            family="soft_controls",
            components=["soft_reason"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="soft_reason",
            note="Tests a weak reasoning cue that is less intrusive than step-by-step prompting.",
        ),
        build_record(
            "principle2_soft_reason_format",
            "Pause briefly to think before answering, keep the response focused, and end with: FINAL ANSWER: <answer>",
            family="soft_controls",
            components=["soft_reason", "format"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="soft_reason_format",
            note="Tests whether a soft reasoning cue becomes useful when paired with a simple answer contract.",
        ),
        build_record(
            "principle2_hard_reason",
            "Think step by step before answering.",
            family="hard_controls",
            components=["hard_reason"],
            complexity_level="minimal",
            hypothesis_role="control",
            contrast_group="hard_reason",
            note="Control for the earlier strong step-by-step instruction that looked harmful.",
        ),
        build_record(
            "principle2_strong_expert",
            "You are a knowledgeable expert. Answer the question accurately.",
            family="hard_controls",
            components=["strong_expert"],
            complexity_level="minimal",
            hypothesis_role="control",
            contrast_group="strong_expert",
            note="Control for an explicit expert persona without other lightweight boundary-setting cues.",
        ),
        build_record(
            "principle2_verbose_control",
            "You are a world-class interdisciplinary expert with broad knowledge across mathematics, commonsense reasoning, and language understanding. Carefully analyze the task, consider the relevant facts, reason step by step, and maintain a high standard of rigor throughout your solution. Organize your thinking clearly, avoid unnecessary assumptions, and be precise in your final response. End with: FINAL ANSWER: <answer>",
            family="heavy_controls",
            components=["verbose", "strong_expert", "hard_reason", "format"],
            complexity_level="heavy",
            hypothesis_role="control",
            contrast_group="verbose_control",
            note="Heavy verbose control that keeps adding global steering and world-building.",
        ),
        build_record(
            "principle2_multiagent_control",
            "You are a coordinator working with a Solver, a Critic, and a Verifier. First assign roles, then let the Solver reason step by step, let the Critic challenge the reasoning, and let the Verifier check the answer before you respond. End with: FINAL ANSWER: <answer>",
            family="heavy_controls",
            components=["multiagent", "hard_reason", "verify", "format"],
            complexity_level="heavy",
            hypothesis_role="control",
            contrast_group="multiagent_control",
            note="Heavy multi-agent control that tests whether explicit orchestration hurts transfer.",
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
    print(f"[build_principle_refined_prompt_pool] wrote {len(records)} prompts to {output_path}", flush=True)


if __name__ == "__main__":
    main()
