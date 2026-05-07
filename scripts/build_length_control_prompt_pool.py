from __future__ import annotations

import argparse
import json
from pathlib import Path

from _bootstrap import bootstrap_project_root

bootstrap_project_root()


def prompt_record(
    *,
    prompt_id: str,
    text: str,
    role: str,
    components: list[str],
    source_note: str,
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
        "provenance": "controlled_prompt_design",
        "prompt_role": "system",
        "original_prompt_role": "system",
        "task_scope": "task_agnostic",
        "optimized_for_tasks": [],
        "source_datasets": [],
        "source_url": "",
        "paper_title": "",
        "paper_url": "",
    }


PROMPTS = [
    prompt_record(
        prompt_id="lc_plain",
        text="Answer the question accurately.",
        role="plain_control",
        components=[],
        source_note="Neutral controlled prompt used as the length/boundary baseline.",
    ),
    prompt_record(
        prompt_id="lc_length_only_short",
        text="Keep your answer short.",
        role="length_only",
        components=["short"],
        source_note="Shortness instruction without an explicit answer boundary.",
    ),
    prompt_record(
        prompt_id="lc_length_only_one_sentence",
        text="Answer in one short sentence.",
        role="length_only",
        components=["short"],
        source_note="Length-only instruction with no final-answer contract.",
    ),
    prompt_record(
        prompt_id="lc_concise_only",
        text="Give only the final answer.",
        role="concise_only",
        components=["concise"],
        source_note="Concise instruction without an explicit final-answer label.",
    ),
    prompt_record(
        prompt_id="lc_format_only",
        text="End your response with: FINAL ANSWER: <answer>",
        role="boundary_only",
        components=["format"],
        source_note="Explicit answer-boundary contract without a brevity instruction.",
    ),
    prompt_record(
        prompt_id="lc_boundary_only",
        text="Make the final answer easy to identify by writing it after FINAL ANSWER:.",
        role="boundary_only",
        components=["format"],
        source_note="Boundary-only instruction that does not require short output.",
    ),
    prompt_record(
        prompt_id="lc_concise_boundary",
        text="Give only the final answer. Format it as: FINAL ANSWER: <answer>",
        role="concise_boundary",
        components=["concise", "format"],
        source_note="Combines brevity with an explicit answer-boundary contract.",
    ),
    prompt_record(
        prompt_id="lc_careful_boundary",
        text="Read the question carefully and end with: FINAL ANSWER: <answer>",
        role="careful_boundary",
        components=["careful", "format"],
        source_note="Boundary contract with careful-reading cue but no brevity requirement.",
    ),
    prompt_record(
        prompt_id="lc_verbose_boundary",
        text=(
            "You may explain briefly if needed, but clearly mark the final answer "
            "as: FINAL ANSWER: <answer>"
        ),
        role="verbose_boundary",
        components=["verbose", "format"],
        source_note="Allows extra explanation while preserving a final-answer boundary.",
    ),
    prompt_record(
        prompt_id="lc_verbose_no_boundary",
        text="Explain your answer in detail before giving your conclusion.",
        role="verbose_no_boundary",
        components=["verbose"],
        source_note="Verbose control without a final-answer boundary.",
    ),
    prompt_record(
        prompt_id="lc_short_no_boundary_careful",
        text="Read carefully and answer very briefly.",
        role="length_only_careful",
        components=["careful", "short"],
        source_note="Short careful-reading control without explicit final-answer formatting.",
    ),
    prompt_record(
        prompt_id="lc_boundary_no_short_careful",
        text="Read carefully. Put your final response after FINAL ANSWER:.",
        role="boundary_only_careful",
        components=["careful", "format"],
        source_note="Careful boundary control without explicit shortness.",
    ),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build controlled prompts for length-vs-answer-boundary disentanglement."
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
