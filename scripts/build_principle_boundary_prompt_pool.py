from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, resolve_path


PROMPT_REPORT = {
    "source_url": "https://arxiv.org/abs/2406.06608",
    "paper_title": "The Prompt Report: A Systematic Survey of Prompt Engineering Techniques",
    "paper_url": "https://arxiv.org/abs/2406.06608",
}
PROMPTBENCH = {
    "source_url": "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompt_engineering/base.py",
    "paper_title": "PromptBench: A Unified Library for Evaluation of Large Language Models",
    "paper_url": "https://arxiv.org/abs/2312.07910",
}
ZEROSHOT_COT = {
    "source_url": "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompt_engineering/chain_of_thought.py",
    "paper_title": "Large Language Models are Zero-Shot Reasoners",
    "paper_url": "https://arxiv.org/abs/2205.11916",
}
BBH = {
    "source_url": "https://huggingface.co/datasets/Joschka/big_bench_hard",
    "paper_title": "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
    "paper_url": "https://arxiv.org/abs/2210.09261",
}
SELF_CHECK = {
    "source_url": "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompts/method_oriented.py",
    "paper_title": "EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus",
    "paper_url": "https://arxiv.org/abs/2307.11760",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ref-aligned boundary principle probes."
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
    reference: dict[str, str],
):
    return {
        "id": prompt_id,
        "group_id": prompt_id,
        "variant": "original",
        "source": "principle_probe_boundary",
        "source_title": f"Boundary principle probe: {prompt_id}",
        "source_url": reference["source_url"],
        "paper_title": reference["paper_title"],
        "paper_url": reference["paper_url"],
        "provenance": "paper_backed_controlled_probe",
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
            "Answer the question.",
            family="baseline",
            components=[],
            complexity_level="minimal",
            hypothesis_role="baseline",
            contrast_group="plain",
            note="Neutral direct-answer baseline aligned with BBH answer-only prompt format.",
            reference=BBH,
        ),
        build_record(
            "principle3_concise",
            "Answer the question. Give only the final answer.",
            family="minimal_answer_mode",
            components=["concise"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="concise",
            note="Uses the BBH answer-only wording as the source-backed concise/direct-answer control.",
            reference=BBH,
        ),
        build_record(
            "principle3_careful",
            "Read the question carefully. Answer the question.",
            family="instruction_quality",
            components=["careful"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="careful",
            note="Minimal careful-reading variant of the direct-answer baseline, motivated by prompt-instruction clarity controls in prompt-engineering taxonomies.",
            reference=PROMPT_REPORT,
        ),
        build_record(
            "principle3_check",
            "Answer the question. Are you sure that's your final answer? It might be worth taking another look.",
            family="verification",
            components=["check"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="check",
            note="Uses the PromptBench/EmotionPrompt final-answer recheck cue as a source-backed lightweight verification control.",
            reference=SELF_CHECK,
        ),
        build_record(
            "principle3_concise_format",
            "Answer the question. Give only the final answer. Please output your answer at the end as FINAL ANSWER: <answer>.",
            family="minimal_answer_mode",
            components=["concise", "format"],
            complexity_level="minimal",
            hypothesis_role="main_hypothesis",
            contrast_group="concise_format",
            note="Combines BBH answer-only wording with a PromptBench-style answer-format suffix normalized to the repository's FINAL ANSWER marker.",
            reference=PROMPTBENCH,
        ),
        build_record(
            "principle3_careful_format",
            "Read the question carefully. Please output your answer at the end as FINAL ANSWER: <answer>.",
            family="structured_output",
            components=["careful", "format"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_format",
            note="Careful-reading variant paired with a PromptBench-style answer-format suffix normalized to FINAL ANSWER.",
            reference=PROMPTBENCH,
        ),
        build_record(
            "principle3_careful_check",
            "Read the question carefully. Answer the question. Are you sure that's your final answer? It might be worth taking another look.",
            family="verification",
            components=["careful", "check"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_check",
            note="Combines careful reading with the source-backed PromptBench/EmotionPrompt recheck cue.",
            reference=SELF_CHECK,
        ),
        build_record(
            "principle3_careful_format_check",
            "Read the question carefully. Please output your answer at the end as FINAL ANSWER: <answer>. Are you sure that's your final answer? It might be worth taking another look.",
            family="verification",
            components=["careful", "format", "check"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="careful_format_check",
            note="Structured-output control plus the source-backed PromptBench/EmotionPrompt recheck cue.",
            reference=SELF_CHECK,
        ),
        build_record(
            "principle3_soft_reason",
            "Let's think step by step.",
            family="reasoning_elicitation",
            components=["soft_reason"],
            complexity_level="minimal",
            hypothesis_role="positive_probe",
            contrast_group="soft_reason",
            note="Uses the canonical Zero-shot-CoT trigger.",
            reference=ZEROSHOT_COT,
        ),
        build_record(
            "principle3_soft_reason_format",
            "Let's think step by step. Please output your answer at the end as FINAL ANSWER: <answer>.",
            family="reasoning_elicitation",
            components=["soft_reason", "format"],
            complexity_level="lightweight",
            hypothesis_role="positive_probe",
            contrast_group="soft_reason_format",
            note="Combines the canonical Zero-shot-CoT trigger with a PromptBench-style answer-format suffix normalized to FINAL ANSWER.",
            reference=ZEROSHOT_COT,
        ),
        build_record(
            "principle3_concise_careful_format",
            "Read the question carefully. Answer the question. Give only the final answer. Please output your answer at the end as FINAL ANSWER: <answer>.",
            family="minimal_answer_mode",
            components=["concise", "careful", "format"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="concise_careful_format",
            note="Controlled combination of careful reading, BBH answer-only wording, and PromptBench-style answer formatting.",
            reference=PROMPTBENCH,
        ),
        build_record(
            "principle3_concise_careful_check",
            "Read the question carefully. Answer the question. Give only the final answer. Are you sure that's your final answer? It might be worth taking another look.",
            family="minimal_answer_mode",
            components=["concise", "careful", "check"],
            complexity_level="lightweight",
            hypothesis_role="main_hypothesis",
            contrast_group="concise_careful_check",
            note="Controlled combination of BBH answer-only wording and the PromptBench/EmotionPrompt recheck cue.",
            reference=SELF_CHECK,
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
    print(
        f"[build_principle_boundary_prompt_pool] wrote {len(records)} prompts to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
