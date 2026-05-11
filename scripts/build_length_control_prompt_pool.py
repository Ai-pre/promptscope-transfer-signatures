from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()


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
BBH = {
    "source_url": "https://huggingface.co/datasets/Joschka/big_bench_hard",
    "paper_title": "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
    "paper_url": "https://arxiv.org/abs/2210.09261",
}
PROMPTWIZARD = {
    "source_url": "https://raw.githubusercontent.com/microsoft/PromptWizard/main/demos/gsm8k/configs/promptopt_config.yaml",
    "paper_title": "PromptWizard: Task-Aware Prompt Optimization Framework",
    "paper_url": "https://arxiv.org/abs/2405.18369",
}


def prompt_record(
    *,
    prompt_id: str,
    text: str,
    role: str,
    components: list[str],
    source_note: str,
    reference: dict[str, str],
):
    return {
        "id": prompt_id,
        "group_id": prompt_id,
        "variant": "original",
        "source": "length_control",
        "text": text,
        "principle_components_json": json.dumps(components, ensure_ascii=False),
        "length_control_role": role,
        "source_title": f"Length/boundary control: {prompt_id}",
        "source_note": source_note,
        "provenance": "paper_backed_controlled_probe",
        "prompt_role": "system",
        "original_prompt_role": "system",
        "task_scope": "task_agnostic",
        "optimized_for_tasks": [],
        "source_datasets": [],
        "source_url": reference["source_url"],
        "paper_title": reference["paper_title"],
        "paper_url": reference["paper_url"],
    }


PROMPTS = [
    prompt_record(
        prompt_id="lc_plain",
        text="Answer the question.",
        role="plain_control",
        components=[],
        source_note="Neutral direct-answer baseline aligned with BBH answer-only prompt format.",
        reference=BBH,
    ),
    prompt_record(
        prompt_id="lc_length_only_short",
        text="Give only the final answer.",
        role="length_only",
        components=["short"],
        source_note="Source-backed minimal-output control derived from the BBH answer-only prompt format.",
        reference=BBH,
    ),
    prompt_record(
        prompt_id="lc_length_only_one_sentence",
        text="Answer the question. Give only the final answer.",
        role="length_only",
        components=["short"],
        source_note="Length/minimal-output control using the BBH answer-only wording.",
        reference=BBH,
    ),
    prompt_record(
        prompt_id="lc_concise_only",
        text="Answer the question. Give only the final answer.",
        role="concise_only",
        components=["concise"],
        source_note="Answer-only / direct-answer control using the BBH answer-only wording.",
        reference=BBH,
    ),
    prompt_record(
        prompt_id="lc_format_only",
        text="Please output your answer at the end as FINAL ANSWER: <answer>.",
        role="boundary_only",
        components=["format"],
        source_note="PromptBench-style answer-format suffix normalized from ##<answer> to this repository's FINAL ANSWER marker.",
        reference=PROMPTBENCH,
    ),
    prompt_record(
        prompt_id="lc_boundary_only",
        text="Please output your answer at the end as FINAL ANSWER: <answer>.",
        role="boundary_only",
        components=["format"],
        source_note="Boundary-only structured-output control based on PromptBench answer-format prompting.",
        reference=PROMPTBENCH,
    ),
    prompt_record(
        prompt_id="lc_concise_boundary",
        text="Answer the question. Give only the final answer. Please output your answer at the end as FINAL ANSWER: <answer>.",
        role="concise_boundary",
        components=["concise", "format"],
        source_note="Combines BBH answer-only wording with a PromptBench-style answer-format suffix.",
        reference=PROMPTBENCH,
    ),
    prompt_record(
        prompt_id="lc_careful_boundary",
        text="Read the question carefully. Please output your answer at the end as FINAL ANSWER: <answer>.",
        role="careful_boundary",
        components=["careful", "format"],
        source_note="Careful-reading variant paired with PromptBench-style answer formatting; carefulness is a controlled instruction-quality axis.",
        reference=PROMPT_REPORT,
    ),
    prompt_record(
        prompt_id="lc_verbose_boundary",
        text=(
            "For each question present the reasoning followed by the correct answer. "
            "Please output your answer at the end as FINAL ANSWER: <answer>."
        ),
        role="verbose_boundary",
        components=["verbose", "format"],
        source_note="Uses PromptWizard-style reasoning-followed-by-answer wording, with the final marker normalized for evaluation.",
        reference=PROMPTWIZARD,
    ),
    prompt_record(
        prompt_id="lc_verbose_no_boundary",
        text="For each question present the reasoning followed by the correct answer.",
        role="verbose_no_boundary",
        components=["verbose"],
        source_note="Verbose/reasoning control based on PromptWizard GSM8K seed wording, without an explicit final-answer marker.",
        reference=PROMPTWIZARD,
    ),
    prompt_record(
        prompt_id="lc_short_no_boundary_careful",
        text="Read the question carefully. Give only the final answer.",
        role="length_only_careful",
        components=["careful", "short"],
        source_note="Careful-reading variant of the BBH answer-only minimal-output control.",
        reference=BBH,
    ),
    prompt_record(
        prompt_id="lc_boundary_no_short_careful",
        text="Read the question carefully. Please output your answer at the end as FINAL ANSWER: <answer>.",
        role="boundary_only_careful",
        components=["careful", "format"],
        source_note="Careful-reading variant of the PromptBench-style structured-output control.",
        reference=PROMPTBENCH,
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build ref-aligned prompts for length-vs-answer-boundary disentanglement."
    )
    parser.add_argument(
        "--output",
        default="data/prompts_length_control_boundary.jsonl",
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in PROMPTS:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"[build_length_control_prompt_pool] wrote {len(PROMPTS)} prompts to {output_path}")


if __name__ == "__main__":
    main()
