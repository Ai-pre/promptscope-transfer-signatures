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
BBH = {
    "source_url": "https://huggingface.co/datasets/Joschka/big_bench_hard",
    "paper_title": "Challenging BIG-Bench Tasks and Whether Chain-of-Thought Can Solve Them",
    "paper_url": "https://arxiv.org/abs/2210.09261",
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


def record(
    prompt_id: str,
    text: str,
    *,
    taxonomy_role: str,
    components: list[str],
    reference: dict[str, str],
    complexity_level: str = "minimal",
    source_note: str,
    provenance: str = "controlled_prompt_design",
):
    return {
        "id": prompt_id,
        "group_id": prompt_id,
        "variant": "original",
        "source": "taxonomy_control_v2",
        "text": text,
        "principle_components_json": json.dumps(components, ensure_ascii=False),
        "taxonomy_role": taxonomy_role,
        "length_control_role": taxonomy_role,
        "source_title": f"Taxonomy v2 control: {prompt_id}",
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
    # Direct / baseline controls.
    record("taxv2_plain_answer", "Answer the question.", taxonomy_role="direct_baseline", components=[], reference=BBH, source_note="Neutral direct-answer baseline aligned with BBH-style answer-only evaluation.", provenance="paper_backed_controlled_probe"),
    record("taxv2_direct_accurate", "Answer the question accurately.", taxonomy_role="direct_baseline", components=["direct"], reference=PROMPT_REPORT, source_note="Minimal accuracy-oriented direct-answer instruction."),
    record("taxv2_direct_best", "Choose the best answer to the question.", taxonomy_role="direct_baseline", components=["direct"], reference=PROMPT_REPORT, source_note="Minimal best-answer direct instruction."),
    record("taxv2_direct_no_extra", "Answer directly without extra commentary.", taxonomy_role="direct_baseline", components=["direct", "short"], reference=PROMPT_REPORT, source_note="Direct-answer control that also suppresses extra commentary."),
    # Brevity / answer-only controls.
    record("taxv2_keep_answer_short", "Keep your answer short.", taxonomy_role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Brevity control motivated by concise response constraints."),
    record("taxv2_final_answer_only", "Give only the final answer.", taxonomy_role="answer_only", components=["concise", "short"], reference=BBH, source_note="Minimal answer-only wording derived from BBH-style direct-answer prompting.", provenance="paper_backed_controlled_probe"),
    record("taxv2_few_words", "Answer in as few words as possible.", taxonomy_role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Strong brevity control."),
    record("taxv2_one_sentence", "Answer in one sentence.", taxonomy_role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="One-sentence output-length control."),
    record("taxv2_no_explanation", "Give the answer without explanation.", taxonomy_role="answer_only", components=["concise", "short"], reference=BBH, source_note="Answer-only control that explicitly suppresses explanation.", provenance="paper_backed_controlled_probe"),
    record("taxv2_concise_accurate", "Be concise and accurate.", taxonomy_role="brevity", components=["short", "careful"], reference=PROMPT_REPORT, source_note="Concise accuracy-oriented control."),
    record("taxv2_minimal_response", "Use the minimal response needed to answer correctly.", taxonomy_role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Minimal-response control."),
    record("taxv2_answer_only_bbh", "Answer the question. Give only the final answer.", taxonomy_role="answer_only", components=["concise", "short"], reference=BBH, source_note="BBH answer-only wording as a source-backed concise control.", provenance="paper_backed_controlled_probe"),
    # Structured output / formatting controls.
    record("taxv2_final_answer_marker", "Please output your answer at the end as FINAL ANSWER: <answer>.", taxonomy_role="structured_output", components=["format"], reference=PROMPTBENCH, source_note="PromptBench-style answer-format suffix normalized to FINAL ANSWER.", provenance="paper_backed_controlled_probe"),
    record("taxv2_answer_label", "End your response with Answer: <answer>.", taxonomy_role="structured_output", components=["format"], reference=PROMPTBENCH, source_note="Answer-label structured-output control."),
    record("taxv2_final_line_only", "Put the final answer on its own final line.", taxonomy_role="structured_output", components=["format"], reference=PROMPTBENCH, source_note="Final-line formatting control."),
    record("taxv2_option_letter_only", "If choices are provided, answer with only the option letter.", taxonomy_role="structured_output", components=["format", "concise"], reference=PROMPTBENCH, source_note="Choice-format control for multiple-choice extraction."),
    record("taxv2_no_markdown", "Provide the final answer plainly, without markdown or formatting.", taxonomy_role="structured_output", components=["format", "short"], reference=PROMPT_REPORT, source_note="Plain-format output control."),
    record("taxv2_json_answer", "Respond using exactly this JSON format: {\"answer\": \"<answer>\"}.", taxonomy_role="structured_output", components=["format"], reference=PROMPT_REPORT, source_note="Structured JSON output control."),
    # Instruction quality / careful reading controls.
    record("taxv2_read_carefully", "Read the question carefully before answering.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Careful-reading instruction-quality control."),
    record("taxv2_attend_constraints", "Pay close attention to all constraints in the question.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Constraint-attention instruction control."),
    record("taxv2_avoid_overlooking", "Do not overlook details in the question.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Detail-attention control."),
    record("taxv2_verify_question_type", "Identify what the question is asking before answering.", taxonomy_role="instruction_quality", components=["careful", "strategy"], reference=PROMPT_REPORT, source_note="Instruction-quality control focused on task recognition."),
    record("taxv2_be_precise", "Be precise and avoid unnecessary assumptions.", taxonomy_role="instruction_quality", components=["careful"], reference=PROMPT_REPORT, source_note="Precision-oriented instruction control."),
    record("taxv2_interpret_before_answer", "Interpret the question first, then answer clearly.", taxonomy_role="instruction_quality", components=["careful", "strategy"], reference=PROMPT_REPORT, source_note="Lightweight interpret-then-answer control."),
    # Verification / self-check controls.
    record("taxv2_self_check", "Answer the question. Are you sure that's your final answer? It might be worth taking another look.", taxonomy_role="verification", components=["check"], reference=SELF_CHECK, source_note="PromptBench/EmotionPrompt self-check wording.", provenance="paper_backed_controlled_probe"),
    record("taxv2_double_check", "Double-check your answer before responding.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Self-verification control."),
    record("taxv2_check_constraints", "Check that your answer satisfies the question constraints.", taxonomy_role="verification", components=["check", "careful"], reference=PROMPT_REPORT, source_note="Constraint-verification control."),
    record("taxv2_reconsider_uncertain", "If uncertain, reconsider the answer before finalizing it.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Uncertainty-triggered verification control."),
    record("taxv2_avoid_mistakes", "Avoid common mistakes and answer carefully.", taxonomy_role="verification", components=["check", "careful"], reference=PROMPT_REPORT, source_note="Mistake-avoidance verification control."),
    record("taxv2_final_sanity_check", "Perform a quick sanity check, then give the answer.", taxonomy_role="verification", components=["check"], reference=PROMPT_REPORT, source_note="Final sanity-check control."),
    # Reasoning / decomposition controls.
    record("taxv2_zero_shot_cot", "Let's think step by step.", taxonomy_role="reasoning_elicitation", components=["hard_reason"], reference=ZERO_SHOT_COT, source_note="Canonical zero-shot CoT trigger.", provenance="paper_backed_controlled_probe", complexity_level="heavy"),
    record("taxv2_step_by_step_final", "Think step by step, then give the final answer.", taxonomy_role="reasoning_elicitation", components=["hard_reason"], reference=ZERO_SHOT_COT, source_note="Zero-shot CoT with explicit final-answer request.", complexity_level="heavy"),
    record("taxv2_brief_reasoning", "Use brief reasoning to arrive at the answer.", taxonomy_role="reasoning_elicitation", components=["soft_reason"], reference=ZERO_SHOT_COT, source_note="Soft reasoning cue without full CoT pressure.", complexity_level="lightweight"),
    record("taxv2_solve_systematically", "Solve the problem systematically.", taxonomy_role="reasoning_elicitation", components=["soft_reason", "strategy"], reference=PROMPT_REPORT, source_note="Systematic-problem-solving control.", complexity_level="lightweight"),
    record("taxv2_decompose_subproblems", "Break the problem into smaller subproblems before answering.", taxonomy_role="decomposition", components=["decomposition", "hard_reason"], reference=LEAST_TO_MOST, source_note="Problem decomposition control inspired by least-to-most prompting.", complexity_level="heavy"),
    record("taxv2_least_to_most", "Solve the simplest part first, then use it to solve the full problem.", taxonomy_role="decomposition", components=["decomposition", "hard_reason"], reference=LEAST_TO_MOST, source_note="Least-to-most style decomposition control.", complexity_level="heavy"),
    record("taxv2_plan_then_answer", "Make a brief plan, then answer.", taxonomy_role="reasoning_elicitation", components=["strategy", "soft_reason"], reference=PROMPT_REPORT, source_note="Plan-then-answer control.", complexity_level="lightweight"),
    record("taxv2_calculate_stepwise", "For calculation questions, compute step by step before answering.", taxonomy_role="reasoning_elicitation", components=["hard_reason"], reference=ZERO_SHOT_COT, source_note="Calculation-specific stepwise reasoning control.", complexity_level="heavy"),
    # Knowledge / meta / strategy controls.
    record("taxv2_generated_knowledge", "Generate some knowledge about the input, then answer the question.", taxonomy_role="knowledge_elicitation", components=["knowledge"], reference=GENERATED_KNOWLEDGE, source_note="Generated-knowledge prompting control.", complexity_level="heavy"),
    record("taxv2_relevant_facts", "Identify the relevant facts, then answer.", taxonomy_role="knowledge_elicitation", components=["knowledge", "strategy"], reference=GENERATED_KNOWLEDGE, source_note="Relevant-fact elicitation control.", complexity_level="lightweight"),
    record("taxv2_task_structure", "Focus on the structure of the task before answering.", taxonomy_role="meta_prompting", components=["meta_prompting", "strategy"], reference=META_PROMPTING, source_note="Meta-prompting control focused on task structure.", complexity_level="lightweight"),
    record("taxv2_choose_strategy", "Choose an appropriate strategy for the task, then answer.", taxonomy_role="strategy", components=["strategy"], reference=PROMPT_REPORT, source_note="Task-strategy selection control.", complexity_level="lightweight"),
    record("taxv2_meta_prompting_structure", "Focus on the formal structure of the task rather than content-specific examples, then answer.", taxonomy_role="meta_prompting", components=["meta_prompting", "strategy"], reference=META_PROMPTING, source_note="Meta Prompting structure-oriented control.", complexity_level="heavy"),
    record("taxv2_abstract_strategy", "Use general problem-solving principles rather than memorized examples.", taxonomy_role="meta_prompting", components=["meta_prompting", "strategy"], reference=META_PROMPTING, source_note="Abstract strategy control.", complexity_level="lightweight"),
    record("taxv2_promptwizard_like", "Use the task intent, a suitable expert perspective, and a concise reasoning strategy to produce the best answer.", taxonomy_role="promptwizard_like", components=["prompt_optimization", "expert", "strategy", "soft_reason"], reference=PROMPTWIZARD, source_note="PromptWizard-like task-aware prompt optimization control.", complexity_level="heavy"),
    record("taxv2_general_problem_solving", "Apply a general problem-solving strategy and provide the answer.", taxonomy_role="strategy", components=["strategy"], reference=PROMPT_REPORT, source_note="General strategy control.", complexity_level="lightweight"),
    # Persona / emotion / role controls.
    record("taxv2_expert_persona", "You are a distinguished expert in the relevant domain. Answer the question accurately.", taxonomy_role="expert_persona", components=["expert"], reference=EXPERT_PROMPTING, source_note="Expert persona prompting control.", complexity_level="lightweight"),
    record("taxv2_domain_expert", "Act as a domain expert and answer accurately.", taxonomy_role="expert_persona", components=["expert"], reference=EXPERT_PROMPTING, source_note="Compact expert-role control.", complexity_level="minimal"),
    record("taxv2_meticulous_expert", "You are a meticulous expert. Be accurate and concise.", taxonomy_role="expert_persona", components=["expert", "careful", "short"], reference=EXPERT_PROMPTING, source_note="Expert role combined with careful brevity.", complexity_level="lightweight"),
    record("taxv2_career_importance", "This is important to my career. Please give your best and most accurate answer.", taxonomy_role="emotional_stimulus", components=["emotion"], reference=EMOTION_PROMPT, source_note="Representative EmotionPrompt-style stimulus.", complexity_level="lightweight"),
    record("taxv2_high_stakes_accuracy", "The correctness of this answer matters a lot. Be careful.", taxonomy_role="emotional_stimulus", components=["emotion", "careful"], reference=EMOTION_PROMPT, source_note="High-stakes emotional stimulus control.", complexity_level="lightweight"),
    record("taxv2_confident_calm", "Stay calm and confident while answering accurately.", taxonomy_role="emotional_stimulus", components=["emotion"], reference=EMOTION_PROMPT, source_note="Calm-confidence affective control.", complexity_level="lightweight"),
    record("taxv2_diligent_assistant", "You are a diligent assistant. Think carefully and answer accurately.", taxonomy_role="role_behavioral", components=["role", "careful"], reference=SPRIG, source_note="SPRIG-style system role and behavioral-property control.", complexity_level="lightweight"),
    record("taxv2_professional_exam", "Treat this like a professional exam question and answer carefully.", taxonomy_role="scenario", components=["scenario", "careful"], reference=PROMPT_REPORT, source_note="Scenario framing control.", complexity_level="lightweight"),
    # SPRIG-style mixed component controls.
    record("taxv2_careful_short", "Read carefully and give only the final answer.", taxonomy_role="mixed_brevity_careful", components=["careful", "concise", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: carefulness plus brevity.", complexity_level="lightweight"),
    record("taxv2_expert_short", "As an expert, give the shortest correct answer.", taxonomy_role="mixed_expert_brevity", components=["expert", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: expert role plus brevity.", complexity_level="lightweight"),
    record("taxv2_check_short", "Double-check, then give only the final answer.", taxonomy_role="mixed_check_brevity", components=["check", "concise", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: verification plus answer-only response.", complexity_level="lightweight"),
    record("taxv2_structure_short", "Use the task structure to answer as briefly as possible.", taxonomy_role="mixed_meta_brevity", components=["meta_prompting", "strategy", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: structure awareness plus brevity.", complexity_level="lightweight"),
    record("taxv2_knowledge_short", "Use relevant knowledge and give only the answer.", taxonomy_role="mixed_knowledge_brevity", components=["knowledge", "concise", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: knowledge elicitation plus answer-only response.", complexity_level="lightweight"),
    record("taxv2_meta_short", "Focus on the task structure and answer concisely.", taxonomy_role="mixed_meta_brevity", components=["meta_prompting", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: meta prompting plus concise answer.", complexity_level="lightweight"),
    record("taxv2_careful_final_marker", "Read carefully, then end with FINAL ANSWER: <answer>.", taxonomy_role="mixed_careful_format", components=["careful", "format"], reference=SPRIG, source_note="SPRIG-style mixed component: carefulness plus structured output.", complexity_level="lightweight"),
    record("taxv2_emotion_short", "This matters a lot. Give the shortest accurate answer.", taxonomy_role="mixed_emotion_brevity", components=["emotion", "short"], reference=SPRIG, source_note="SPRIG-style mixed component: emotional stimulus plus brevity.", complexity_level="lightweight"),
    record("taxv2_strategy_check", "Choose a strategy, check it briefly, then answer.", taxonomy_role="mixed_strategy_check", components=["strategy", "check"], reference=SPRIG, source_note="SPRIG-style mixed component: strategy plus verification.", complexity_level="lightweight"),
    record("taxv2_expert_check", "As an expert, verify the answer before responding.", taxonomy_role="mixed_expert_check", components=["expert", "check"], reference=SPRIG, source_note="SPRIG-style mixed component: expert role plus verification.", complexity_level="lightweight"),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build expanded taxonomy v2 prompt controls.")
    parser.add_argument("--output", default="data/prompts_taxonomy_v2.jsonl")
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
    print(f"[build_taxonomy_v2_prompt_pool] wrote {len(PROMPTS)} prompts to {output_path}")


if __name__ == "__main__":
    main()
