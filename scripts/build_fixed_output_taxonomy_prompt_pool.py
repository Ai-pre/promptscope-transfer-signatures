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
SPRIG = {
    "source_url": "https://arxiv.org/abs/2410.14826",
    "paper_title": "SPRIG: Improving Large Language Model Performance by System Prompt Optimization",
    "paper_url": "https://arxiv.org/abs/2410.14826",
}
PROMPTBENCH = {
    "source_url": "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompt_engineering/base.py",
    "paper_title": "PromptBench: A Unified Library for Evaluation of Large Language Models",
    "paper_url": "https://arxiv.org/abs/2312.07910",
}
SELF_CHECK = {
    "source_url": "https://raw.githubusercontent.com/microsoft/promptbench/main/promptbench/prompts/method_oriented.py",
    "paper_title": "EmotionPrompt: Leveraging Psychology for Large Language Models Enhancement via Emotional Stimulus",
    "paper_url": "https://arxiv.org/abs/2307.11760",
}
ZERO_SHOT_COT = {
    "source_url": "https://arxiv.org/abs/2205.11916",
    "paper_title": "Large Language Models are Zero-Shot Reasoners",
    "paper_url": "https://arxiv.org/abs/2205.11916",
}
LEAST_TO_MOST = {
    "source_url": "https://arxiv.org/abs/2205.10625",
    "paper_title": "Least-to-Most Prompting Enables Complex Reasoning in Large Language Models",
    "paper_url": "https://arxiv.org/abs/2205.10625",
}
GENERATED_KNOWLEDGE = {
    "source_url": "https://arxiv.org/abs/2110.08387",
    "paper_title": "Generated Knowledge Prompting for Commonsense Reasoning",
    "paper_url": "https://arxiv.org/abs/2110.08387",
}
EXPERT_PROMPTING = {
    "source_url": "https://arxiv.org/abs/2305.14688",
    "paper_title": "ExpertPrompting: Instructing Large Language Models to be Distinguished Experts",
    "paper_url": "https://arxiv.org/abs/2305.14688",
}
EMOTION_PROMPT = {
    "source_url": "https://arxiv.org/abs/2307.11760",
    "paper_title": "Large Language Models Understand and Can be Enhanced by Emotional Stimuli",
    "paper_url": "https://arxiv.org/abs/2307.11760",
}
META_PROMPTING = {
    "source_url": "https://arxiv.org/abs/2311.11482",
    "paper_title": "Meta Prompting for AI Systems",
    "paper_url": "https://arxiv.org/abs/2311.11482",
}
PROMPTWIZARD = {
    "source_url": "https://arxiv.org/abs/2405.18369",
    "paper_title": "PromptWizard: Task-Aware Prompt Optimization Framework",
    "paper_url": "https://arxiv.org/abs/2405.18369",
}


FIXED_OUTPUT_SUFFIX = "Give only the final answer. Do not explain."


def fixed(prefix: str) -> str:
    prefix = prefix.strip()
    if not prefix:
        return FIXED_OUTPUT_SUFFIX
    return f"{prefix} {FIXED_OUTPUT_SUFFIX}"


def record(
    prompt_id: str,
    prefix: str,
    *,
    taxonomy_role: str,
    components: list[str],
    reference: dict[str, str],
    complexity_level: str = "minimal",
    source_note: str,
    provenance: str = "controlled_prompt_design",
):
    components = list(dict.fromkeys([*components, "fixed_output"]))
    return {
        "id": prompt_id,
        "group_id": prompt_id,
        "variant": "original",
        "source": "fixed_output_taxonomy",
        "text": fixed(prefix),
        "fixed_output_suffix": FIXED_OUTPUT_SUFFIX,
        "principle_components_json": json.dumps(components, ensure_ascii=False),
        "taxonomy_role": taxonomy_role,
        "length_control_role": taxonomy_role,
        "source_title": f"Fixed-output taxonomy control: {prompt_id}",
        "source_note": source_note,
        "provenance": provenance,
        "prompt_role": "system",
        "original_prompt_role": "system",
        "task_scope": "task_agnostic",
        "optimized_for_tasks": [],
        "source_datasets": [],
        "source_url": reference["source_url"],
        "paper_title": reference["paper_title"],
        "paper_url": reference["paper_url"],
        "complexity_level": complexity_level,
    }


PROMPTS = [
    record("fo_direct_answer", "Answer the question.", taxonomy_role="direct_baseline", components=["direct"], reference=PROMPT_REPORT, source_note="Direct baseline under a shared final-answer-only output constraint."),
    record("fo_direct_accurate", "Answer the question accurately.", taxonomy_role="direct_baseline", components=["direct"], reference=PROMPT_REPORT, source_note="Accuracy-oriented direct baseline under fixed output."),
    record("fo_direct_best", "Choose the best answer to the question.", taxonomy_role="direct_baseline", components=["direct"], reference=PROMPT_REPORT, source_note="Best-answer direct baseline under fixed output."),
    record("fo_direct_precise", "Be precise.", taxonomy_role="direct_baseline", components=["direct"], reference=PROMPT_REPORT, source_note="Precision-oriented direct baseline under fixed output."),
    record("fo_careful_read", "Read the question carefully before answering.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Careful-reading instruction-quality control under fixed output."),
    record("fo_careful_constraints", "Pay close attention to every constraint in the question.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Constraint-attention control under fixed output."),
    record("fo_careful_details", "Do not overlook any important detail.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Detail-attention control under fixed output."),
    record("fo_careful_question_type", "Identify what the question is asking.", taxonomy_role="instruction_quality", components=["careful", "strategy"], reference=PROMPT_REPORT, source_note="Task-recognition instruction-quality control under fixed output."),
    record("fo_careful_no_assumptions", "Avoid unnecessary assumptions.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Assumption-avoidance control under fixed output."),
    record("fo_careful_interpret", "Interpret the question carefully.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Question-interpretation control under fixed output."),
    record("fo_check_self", "Are you sure that's your final answer? It might be worth taking another look.", taxonomy_role="verification", components=["check"], reference=SELF_CHECK, source_note="PromptBench/EmotionPrompt self-check cue under fixed output.", provenance="paper_backed_controlled_probe"),
    record("fo_check_double", "Double-check your answer before responding.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Self-verification control under fixed output."),
    record("fo_check_constraints", "Check that the answer satisfies the question constraints.", taxonomy_role="verification", components=["check", "careful"], reference=PROMPT_REPORT, source_note="Constraint-checking control under fixed output."),
    record("fo_check_uncertain", "If uncertain, reconsider before finalizing.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Uncertainty-triggered checking under fixed output."),
    record("fo_check_mistakes", "Avoid common mistakes.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Mistake-avoidance control under fixed output."),
    record("fo_check_sanity", "Perform a quick sanity check internally.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Sanity-check control under fixed output."),
    record("fo_soft_reason_brief", "Think briefly before answering.", taxonomy_role="soft_reasoning", components=["soft_reason"], reference=ZERO_SHOT_COT, source_note="Soft reasoning cue under fixed output; reasoning should not be verbalized.", complexity_level="lightweight"),
    record("fo_soft_reason_systematic", "Solve the problem systematically.", taxonomy_role="soft_reasoning", components=["soft_reason", "strategy"], reference=PROMPT_REPORT, source_note="Systematic soft-reasoning control under fixed output.", complexity_level="lightweight"),
    record("fo_soft_reason_plan", "Make a brief internal plan before answering.", taxonomy_role="soft_reasoning", components=["soft_reason", "strategy"], reference=PROMPT_REPORT, source_note="Internal plan cue under fixed output.", complexity_level="lightweight"),
    record("fo_soft_reason_calculate", "For calculation questions, compute internally before answering.", taxonomy_role="soft_reasoning", components=["soft_reason"], reference=ZERO_SHOT_COT, source_note="Internal calculation cue under fixed output.", complexity_level="lightweight"),
    record("fo_soft_reason_compare", "Compare the plausible options internally.", taxonomy_role="soft_reasoning", components=["soft_reason", "strategy"], reference=PROMPT_REPORT, source_note="Internal option-comparison cue under fixed output.", complexity_level="lightweight"),
    record("fo_soft_reason_minimal", "Use minimal internal reasoning.", taxonomy_role="soft_reasoning", components=["soft_reason"], reference=ZERO_SHOT_COT, source_note="Minimal internal reasoning cue under fixed output.", complexity_level="lightweight"),
    record("fo_hard_reason_step", "Let's think step by step internally.", taxonomy_role="hard_reasoning", components=["hard_reason"], reference=ZERO_SHOT_COT, source_note="Zero-shot-CoT style stepwise reasoning pressure under fixed output.", provenance="paper_backed_controlled_probe", complexity_level="heavy"),
    record("fo_hard_reason_chain", "Work through the reasoning chain internally.", taxonomy_role="hard_reasoning", components=["hard_reason"], reference=ZERO_SHOT_COT, source_note="Chain-of-thought pressure under fixed output.", complexity_level="heavy"),
    record("fo_decompose_subproblems", "Break the problem into smaller subproblems internally.", taxonomy_role="decomposition", components=["decomposition", "hard_reason"], reference=LEAST_TO_MOST, source_note="Least-to-most style decomposition cue under fixed output.", complexity_level="heavy"),
    record("fo_decompose_simple_first", "Solve the simplest part first internally.", taxonomy_role="decomposition", components=["decomposition", "hard_reason"], reference=LEAST_TO_MOST, source_note="Simple-to-complex decomposition cue under fixed output.", complexity_level="heavy"),
    record("fo_hard_reason_verify_steps", "Reason through the steps and verify them internally.", taxonomy_role="hard_reasoning", components=["hard_reason", "check"], reference=ZERO_SHOT_COT, source_note="Stepwise reasoning plus checking under fixed output.", complexity_level="heavy"),
    record("fo_hard_reason_explain_to_self", "Explain the solution to yourself internally.", taxonomy_role="hard_reasoning", components=["hard_reason"], reference=ZERO_SHOT_COT, source_note="Private explanation cue under fixed output.", complexity_level="heavy"),
    record("fo_strategy_choose", "Choose an appropriate strategy for the task.", taxonomy_role="strategy", components=["strategy"], reference=PROMPT_REPORT, source_note="Strategy-selection control under fixed output."),
    record("fo_strategy_structure", "Focus on the structure of the task.", taxonomy_role="strategy", components=["strategy", "meta_prompting"], reference=META_PROMPTING, source_note="Task-structure strategy control under fixed output."),
    record("fo_strategy_general", "Apply general problem-solving principles.", taxonomy_role="strategy", components=["strategy"], reference=PROMPT_REPORT, source_note="General strategy control under fixed output."),
    record("fo_strategy_abstract", "Use abstract reasoning rather than memorized examples.", taxonomy_role="strategy", components=["strategy", "meta_prompting"], reference=META_PROMPTING, source_note="Abstract strategy control under fixed output."),
    record("fo_strategy_task_intent", "Infer the task intent before answering.", taxonomy_role="strategy", components=["strategy"], reference=PROMPT_REPORT, source_note="Task-intent strategy control under fixed output."),
    record("fo_meta_formal", "Focus on the formal structure of the task.", taxonomy_role="meta_prompting", components=["meta_prompting"], reference=META_PROMPTING, source_note="Meta-prompting structure control under fixed output."),
    record("fo_knowledge_relevant", "Identify relevant facts internally.", taxonomy_role="knowledge_elicitation", components=["knowledge"], reference=GENERATED_KNOWLEDGE, source_note="Generated-knowledge control under fixed output."),
    record("fo_knowledge_background", "Recall useful background knowledge internally.", taxonomy_role="knowledge_elicitation", components=["knowledge"], reference=GENERATED_KNOWLEDGE, source_note="Background-knowledge control under fixed output."),
    record("fo_knowledge_commonsense", "Use relevant commonsense knowledge.", taxonomy_role="knowledge_elicitation", components=["knowledge"], reference=GENERATED_KNOWLEDGE, source_note="Commonsense knowledge cue under fixed output."),
    record("fo_knowledge_domain", "Use the relevant domain knowledge.", taxonomy_role="knowledge_elicitation", components=["knowledge"], reference=GENERATED_KNOWLEDGE, source_note="Domain-knowledge cue under fixed output."),
    record("fo_expert_domain", "Act as a domain expert.", taxonomy_role="expert_persona", components=["expert"], reference=EXPERT_PROMPTING, source_note="Expert persona control under fixed output."),
    record("fo_expert_distinguished", "You are a distinguished expert in the relevant domain.", taxonomy_role="expert_persona", components=["expert"], reference=EXPERT_PROMPTING, source_note="Distinguished expert persona under fixed output."),
    record("fo_expert_meticulous", "You are a meticulous expert.", taxonomy_role="expert_persona", components=["expert", "careful"], reference=EXPERT_PROMPTING, source_note="Meticulous expert role under fixed output."),
    record("fo_expert_examiner", "Answer like an expert examiner.", taxonomy_role="expert_persona", components=["expert", "role"], reference=EXPERT_PROMPTING, source_note="Expert examiner role under fixed output."),
    record("fo_expert_specialist", "Use the perspective of a specialist.", taxonomy_role="expert_persona", components=["expert"], reference=EXPERT_PROMPTING, source_note="Specialist perspective control under fixed output."),
    record("fo_emotion_career", "This is important to my career.", taxonomy_role="emotional_stimulus", components=["emotion"], reference=EMOTION_PROMPT, source_note="EmotionPrompt-style career-importance stimulus under fixed output."),
    record("fo_emotion_high_stakes", "The correctness of this answer matters a lot.", taxonomy_role="emotional_stimulus", components=["emotion"], reference=EMOTION_PROMPT, source_note="High-stakes emotional stimulus under fixed output."),
    record("fo_emotion_best_effort", "Please give your best answer.", taxonomy_role="emotional_stimulus", components=["emotion"], reference=EMOTION_PROMPT, source_note="Best-effort emotional stimulus under fixed output."),
    record("fo_emotion_calm", "Stay calm and confident.", taxonomy_role="emotional_stimulus", components=["emotion"], reference=EMOTION_PROMPT, source_note="Calm-confidence affective control under fixed output."),
    record("fo_emotion_attention", "This requires your full attention.", taxonomy_role="emotional_stimulus", components=["emotion", "careful"], reference=EMOTION_PROMPT, source_note="Attention-oriented emotional stimulus under fixed output."),
    record("fo_mix_careful_check", "Read carefully and double-check internally.", taxonomy_role="mixed_careful_check", components=["careful", "check"], reference=SPRIG, source_note="SPRIG-style carefulness plus checking under fixed output.", complexity_level="lightweight"),
    record("fo_mix_expert_careful", "As an expert, read carefully.", taxonomy_role="mixed_expert_careful", components=["expert", "careful"], reference=SPRIG, source_note="SPRIG-style expert plus carefulness under fixed output.", complexity_level="lightweight"),
    record("fo_mix_expert_check", "As an expert, verify internally.", taxonomy_role="mixed_expert_check", components=["expert", "check"], reference=SPRIG, source_note="SPRIG-style expert plus checking under fixed output.", complexity_level="lightweight"),
    record("fo_mix_emotion_careful", "This matters a lot, so read carefully.", taxonomy_role="mixed_emotion_careful", components=["emotion", "careful"], reference=SPRIG, source_note="SPRIG-style emotional stimulus plus carefulness under fixed output.", complexity_level="lightweight"),
    record("fo_mix_strategy_check", "Choose a strategy and sanity-check internally.", taxonomy_role="mixed_strategy_check", components=["strategy", "check"], reference=SPRIG, source_note="SPRIG-style strategy plus checking under fixed output.", complexity_level="lightweight"),
    record("fo_mix_knowledge_strategy", "Use relevant facts and the task structure.", taxonomy_role="mixed_knowledge_strategy", components=["knowledge", "strategy"], reference=SPRIG, source_note="SPRIG-style knowledge plus strategy under fixed output.", complexity_level="lightweight"),
    record("fo_mix_meta_check", "Focus on the task structure and verify internally.", taxonomy_role="mixed_meta_check", components=["meta_prompting", "check"], reference=SPRIG, source_note="SPRIG-style meta prompting plus checking under fixed output.", complexity_level="lightweight"),
    record("fo_mix_promptwizard", "Use task intent, expert perspective, and a suitable internal strategy.", taxonomy_role="promptwizard_like", components=["prompt_optimization", "expert", "strategy"], reference=PROMPTWIZARD, source_note="PromptWizard-like optimized instruction under fixed output.", complexity_level="heavy"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build fixed-output taxonomy prompt controls.")
    parser.add_argument("--output", default="data/prompts_fixed_output_taxonomy.jsonl")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    seen_ids = set()
    with output_path.open("w", encoding="utf-8") as handle:
        for prompt in PROMPTS:
            prompt_id = prompt["id"]
            if prompt_id in seen_ids:
                raise ValueError(f"Duplicate prompt id: {prompt_id}")
            seen_ids.add(prompt_id)
            handle.write(json.dumps(prompt, ensure_ascii=False) + "\n")
    print(f"[build_fixed_output_taxonomy_prompt_pool] wrote {len(PROMPTS)} prompts to {output_path}")


if __name__ == "__main__":
    main()
