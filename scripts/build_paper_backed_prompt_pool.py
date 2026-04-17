from __future__ import annotations

import argparse
import json
import textwrap
import urllib.request

import yaml

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.utils.io import ensure_dir, resolve_path


META_PROMPTING_REPO = "https://github.com/suzgunmirac/meta-prompting"
PROMPTWIZARD_REPO = "https://github.com/microsoft/PromptWizard"
PROMPTBENCH_REPO = "https://github.com/microsoft/promptbench"

META_PROMPTING_PAPER_URL = "https://arxiv.org/abs/2401.12954"
PROMPTWIZARD_PAPER_URL = "https://arxiv.org/abs/2405.18369"
COT_PAPER_URL = "https://arxiv.org/abs/2201.11903"
ZS_COT_PAPER_URL = "https://arxiv.org/abs/2205.11916"
LEAST_TO_MOST_PAPER_URL = "https://arxiv.org/abs/2205.10625"
GENERATED_KNOWLEDGE_PAPER_URL = "https://arxiv.org/abs/2110.08387"
EXPERT_PROMPTING_PAPER_URL = "https://arxiv.org/abs/2305.14688"
EMOTION_PROMPT_PAPER_URL = "https://arxiv.org/abs/2307.11760"

META_PROMPTING_FILES = {
    "meta_prompting_meta_expert": {
        "url": "https://raw.githubusercontent.com/suzgunmirac/meta-prompting/main/prompts/meta-prompting-instruction.txt",
        "source_title": "Meta-Prompting system instruction",
        "source_note": "Official task-agnostic scaffolding prompt from the meta-prompting repository.",
    },
    "meta_prompting_expert_generic": {
        "url": "https://raw.githubusercontent.com/suzgunmirac/meta-prompting/main/prompts/expert-generic-instruction.txt",
        "source_title": "Meta-Prompting expert generic instruction",
        "source_note": "Official generic expert instruction from the meta-prompting repository.",
    },
    "meta_prompting_multipersona": {
        "url": "https://raw.githubusercontent.com/suzgunmirac/meta-prompting/main/prompts/multipersona-prompting-text.txt",
        "source_title": "Meta-Prompting multipersona prompt",
        "source_note": "Official multi-persona collaboration prompt from the meta-prompting repository.",
    },
}

PROMPTWIZARD_CONFIGS = {
    "promptwizard_gsm8k_seed": {
        "task_name": "gsm8k",
        "url": "https://raw.githubusercontent.com/microsoft/PromptWizard/main/demos/gsm8k/configs/promptopt_config.yaml",
        "source_title": "PromptWizard GSM8K seed config",
    },
    "promptwizard_svamp_seed": {
        "task_name": "svamp",
        "url": "https://raw.githubusercontent.com/microsoft/PromptWizard/main/demos/svamp/configs/promptopt_config.yaml",
        "source_title": "PromptWizard SVAMP seed config",
    },
    "promptwizard_aquarat_seed": {
        "task_name": "aqua_rat",
        "url": "https://raw.githubusercontent.com/microsoft/PromptWizard/main/demos/aquarat/configs/promptopt_config.yaml",
        "source_title": "PromptWizard AQUARAT seed config",
    },
}

PROMPTBENCH_METHODS_URL = "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompts/method_oriented.py"
PROMPTBENCH_BASE_URL = "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompt_engineering/base.py"
PROMPTBENCH_CHAIN_URL = "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompt_engineering/chain_of_thought.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a prompt pool from official paper/repository prompt sources."
    )
    parser.add_argument("--output", default="data/prompts_paper_backed.jsonl")
    return parser.parse_args()


def fetch_text(url: str) -> str:
    with urllib.request.urlopen(url, timeout=30) as response:
        return response.read().decode("utf-8")


def normalize_text_block(text: str) -> str:
    return textwrap.dedent(text).strip()


def load_promptbench_method_prompts():
    namespace: dict[str, object] = {}
    exec(fetch_text(PROMPTBENCH_METHODS_URL), namespace)
    prompts = namespace["METHOD_ORIENTED_PROMPTS"]
    if not isinstance(prompts, dict):
        raise ValueError("Could not recover METHOD_ORIENTED_PROMPTS from PromptBench.")
    return prompts


def build_promptwizard_seed_records():
    records = []
    for prompt_id, spec in PROMPTWIZARD_CONFIGS.items():
        config = yaml.safe_load(fetch_text(spec["url"]))
        text = "\n\n".join(
            [
                str(config["task_description"]).strip(),
                str(config["base_instruction"]).strip(),
                str(config["answer_format"]).strip(),
            ]
        )
        task_name = spec["task_name"]
        records.append(
            {
                "id": prompt_id,
                "group_id": prompt_id,
                "variant": "original",
                "source": "promptwizard",
                "source_title": spec["source_title"],
                "source_url": spec["url"],
                "paper_title": "PromptWizard: Task-Aware Prompt Optimization Framework",
                "paper_url": PROMPTWIZARD_PAPER_URL,
                "provenance": "official_repo",
                "prompt_role": "system",
                "original_prompt_role": "system",
                "task_scope": "task_specific",
                "optimized_for_tasks": [task_name],
                "source_datasets": [task_name],
                "source_note": (
                    "Built by concatenating PromptWizard's task_description, "
                    "base_instruction, and answer_format fields from the official demo config."
                ),
                "text": text,
            }
        )
    return records


def build_meta_prompting_records():
    records = []
    for prompt_id, spec in META_PROMPTING_FILES.items():
        records.append(
            {
                "id": prompt_id,
                "group_id": prompt_id,
                "variant": "original",
                "source": "meta_prompting",
                "source_title": spec["source_title"],
                "source_url": spec["url"],
                "paper_title": "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding",
                "paper_url": META_PROMPTING_PAPER_URL,
                "provenance": "official_repo",
                "prompt_role": "system",
                "original_prompt_role": "system",
                "task_scope": "task_agnostic",
                "optimized_for_tasks": [],
                "source_datasets": [],
                "source_note": spec["source_note"],
                "text": normalize_text_block(fetch_text(spec["url"])),
            }
        )
    return records


def build_promptbench_records():
    method_prompts = load_promptbench_method_prompts()
    output_ranges = {
        "gsm8k": "arabic numerals",
        "csqa": "among A through E",
    }

    def make_promptbench_record(
        *,
        prompt_id: str,
        text: str,
        source_url: str,
        source_title: str,
        paper_title: str,
        paper_url: str,
        optimized_for_tasks: list[str],
        original_prompt_role: str,
        source_note: str,
    ):
        task_scope = "task_specific" if optimized_for_tasks else "task_agnostic"
        return {
            "id": prompt_id,
            "group_id": prompt_id,
            "variant": "original",
            "source": "promptbench",
            "source_title": source_title,
            "source_url": source_url,
            "paper_title": paper_title,
            "paper_url": paper_url,
            "provenance": "official_repo",
            "prompt_role": "system",
            "original_prompt_role": original_prompt_role,
            "task_scope": task_scope,
            "optimized_for_tasks": optimized_for_tasks,
            "source_datasets": optimized_for_tasks,
            "source_note": source_note,
            "text": normalize_text_block(text),
        }

    records = [
        make_promptbench_record(
            prompt_id="promptbench_cot_gsm8k",
            text=method_prompts["chain_of_thought"]["gsm8k"],
            source_url=PROMPTBENCH_METHODS_URL,
            source_title="PromptBench CoT few-shot examples for GSM8K",
            paper_title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            paper_url=COT_PAPER_URL,
            optimized_for_tasks=["gsm8k"],
            original_prompt_role="few_shot_user_prefix",
            source_note="Official PromptBench few-shot CoT prefix for GSM8K, rendered as a system prompt in this repository.",
        ),
        make_promptbench_record(
            prompt_id="promptbench_cot_csqa",
            text=method_prompts["chain_of_thought"]["csqa"],
            source_url=PROMPTBENCH_METHODS_URL,
            source_title="PromptBench CoT few-shot examples for CSQA",
            paper_title="Chain-of-Thought Prompting Elicits Reasoning in Large Language Models",
            paper_url=COT_PAPER_URL,
            optimized_for_tasks=["csqa"],
            original_prompt_role="few_shot_user_prefix",
            source_note="Official PromptBench few-shot CoT prefix for CSQA, rendered as a system prompt in this repository.",
        ),
        make_promptbench_record(
            prompt_id="promptbench_least_to_most_gsm8k",
            text=method_prompts["least_to_most"]["gsm8k"],
            source_url=PROMPTBENCH_METHODS_URL,
            source_title="PromptBench least-to-most few-shot examples for GSM8K",
            paper_title="Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
            paper_url=LEAST_TO_MOST_PAPER_URL,
            optimized_for_tasks=["gsm8k"],
            original_prompt_role="few_shot_user_prefix",
            source_note="Official PromptBench least-to-most prompt for GSM8K, rendered as a system prompt in this repository.",
        ),
        make_promptbench_record(
            prompt_id="promptbench_generated_knowledge_csqa",
            text=method_prompts["generated_knowledge"]["csqa"],
            source_url=PROMPTBENCH_METHODS_URL,
            source_title="PromptBench generated-knowledge prompt for CSQA",
            paper_title="Generated Knowledge Prompting for Commonsense Reasoning",
            paper_url=GENERATED_KNOWLEDGE_PAPER_URL,
            optimized_for_tasks=["csqa"],
            original_prompt_role="knowledge_generation_prefix",
            source_note="Official PromptBench generated-knowledge prompt for CSQA, rendered as a system prompt in this repository.",
        ),
        make_promptbench_record(
            prompt_id="promptbench_expert_prompting_meta",
            text=method_prompts["expert_prompting"],
            source_url=PROMPTBENCH_METHODS_URL,
            source_title="PromptBench expert-prompting meta prompt",
            paper_title="ExpertPrompting: Instructing Large Language Models to be Distinguished Experts",
            paper_url=EXPERT_PROMPTING_PAPER_URL,
            optimized_for_tasks=[],
            original_prompt_role="meta_instruction",
            source_note="Official PromptBench expert-prompting meta prompt, rendered as a standalone system prompt in this repository.",
        ),
    ]

    for task_name, output_range in output_ranges.items():
        task_tag = task_name
        records.append(
            make_promptbench_record(
                prompt_id=f"promptbench_zscot_{task_tag}",
                text=(
                    "Let's think step by step.\n"
                    f"Please output your answer at the end as ##<your answer ({output_range})>"
                ),
                source_url=PROMPTBENCH_CHAIN_URL,
                source_title=f"PromptBench zero-shot CoT trigger for {task_name}",
                paper_title="Large Language Models are Zero-Shot Reasoners",
                paper_url=ZS_COT_PAPER_URL,
                optimized_for_tasks=[task_name],
                original_prompt_role="assistant_suffix",
                source_note="Derived from PromptBench's ZSCoT query template (cot trigger plus answer-format suffix), rendered as a system prompt in this repository.",
            )
        )
        records.append(
            make_promptbench_record(
                prompt_id=f"promptbench_baseline_{task_tag}",
                text=f"Please output your answer at the end as ##<your answer ({output_range})>",
                source_url=PROMPTBENCH_BASE_URL,
                source_title=f"PromptBench baseline answer-format prompt for {task_name}",
                paper_title="PromptBench: A Unified Library for Evaluation of Large Language Models",
                paper_url="https://arxiv.org/abs/2312.07910",
                optimized_for_tasks=[task_name],
                original_prompt_role="user_suffix",
                source_note="Derived from PromptBench's Base.query answer-format suffix, rendered as a system prompt in this repository.",
            )
        )

    for prompt_index, prompt_text in enumerate(method_prompts["emotion_prompt"]["prompts"], start=1):
        records.append(
            make_promptbench_record(
                prompt_id=f"promptbench_emotion_prompt_{prompt_index:02d}",
                text=prompt_text,
                source_url=PROMPTBENCH_METHODS_URL,
                source_title=f"PromptBench emotion prompt #{prompt_index}",
                paper_title="EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus",
                paper_url=EMOTION_PROMPT_PAPER_URL,
                optimized_for_tasks=[],
                original_prompt_role="user_suffix",
                source_note="Official PromptBench emotion prompt, rendered as a standalone system prompt in this repository.",
            )
        )

    return records


def dedupe_records(records: list[dict]):
    seen = set()
    unique = []
    for record in records:
        key = record["id"]
        if key in seen:
            raise ValueError(f"Duplicate prompt id detected: {key}")
        seen.add(key)
        unique.append(record)
    return unique


def main():
    args = parse_args()
    project_root = bootstrap_project_root()
    output_path = resolve_path(project_root, args.output)
    ensure_dir(output_path.parent)

    records = []
    records.extend(build_meta_prompting_records())
    records.extend(build_promptwizard_seed_records())
    records.extend(build_promptbench_records())
    records = dedupe_records(records)

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[build_paper_backed_prompt_pool] wrote {len(records)} prompts to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
