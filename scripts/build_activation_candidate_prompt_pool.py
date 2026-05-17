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
ZERO_SHOT_COT = {
    "source_url": "https://arxiv.org/abs/2205.11916",
    "paper_title": "Large Language Models are Zero-Shot Reasoners",
    "paper_url": "https://arxiv.org/abs/2205.11916",
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


def record(
    prompt_id: str,
    text: str,
    *,
    role: str,
    components: list[str],
    reference: dict[str, str],
    source_note: str,
    complexity_level: str = "minimal",
):
    return {
        "id": prompt_id,
        "group_id": prompt_id,
        "variant": "candidate",
        "source": "activation_candidate_pool",
        "text": text,
        "principle_components_json": json.dumps(components, ensure_ascii=False),
        "taxonomy_role": role,
        "length_control_role": role,
        "source_title": f"Activation selection candidate: {prompt_id}",
        "source_note": source_note,
        "provenance": "taxonomy_derived_candidate",
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
    record("cand_answer_only_clean", "Only provide the answer.", role="answer_only", components=["concise", "short"], reference=BBH, source_note="Answer-only candidate derived from BBH-style concise prompting."),
    record("cand_answer_only_no_words", "Return only the answer, with no other words.", role="answer_only", components=["concise", "short"], reference=BBH, source_note="Strict answer-only candidate."),
    record("cand_final_answer_only", "Final answer only.", role="answer_only", components=["concise", "short"], reference=BBH, source_note="Minimal final-answer-only candidate."),
    record("cand_answer_no_explanation", "Give the answer. No explanation.", role="answer_only", components=["concise", "short"], reference=BBH, source_note="Answer-only no-explanation candidate."),
    record("cand_shortest_correct", "Give the shortest correct answer.", role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Brevity candidate."),
    record("cand_fewest_tokens", "Use the fewest words that still answer correctly.", role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Fewest-words candidate."),
    record("cand_compact_answer", "Respond with a compact answer.", role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Compact-response candidate."),
    record("cand_minimal_correct", "Use a minimal correct response.", role="brevity", components=["short"], reference=PROMPT_REPORT, source_note="Minimal-response candidate."),
    record("cand_direct_only", "Answer directly and only answer the question.", role="direct_brevity", components=["direct", "short"], reference=PROMPT_REPORT, source_note="Direct concise candidate."),
    record("cand_direct_no_commentary", "Answer directly. Do not add commentary.", role="direct_brevity", components=["direct", "short"], reference=PROMPT_REPORT, source_note="Direct no-commentary candidate."),
    record("cand_precise_short", "Be precise and brief.", role="direct_brevity", components=["direct", "short", "careful"], reference=PROMPT_REPORT, source_note="Precise brevity candidate."),
    record("cand_exact_short", "Give the exact answer briefly.", role="direct_brevity", components=["direct", "short"], reference=PROMPT_REPORT, source_note="Exact concise answer candidate."),
    record("cand_answer_colon", "End with Answer: <answer>.", role="structured_output", components=["format"], reference=PROMPTBENCH, source_note="Answer-label candidate."),
    record("cand_final_marker_short", "End with FINAL ANSWER: <answer>. Keep it brief.", role="structured_output", components=["format", "short"], reference=PROMPTBENCH, source_note="Final-answer marker plus brevity candidate."),
    record("cand_plain_final_line", "Put only the answer on the final line.", role="structured_output", components=["format", "concise"], reference=PROMPTBENCH, source_note="Final-line answer candidate."),
    record("cand_json_short", "Return {\"answer\": \"<answer>\"} and nothing else.", role="structured_output", components=["format", "short"], reference=PROMPT_REPORT, source_note="JSON answer-only candidate."),
    record("cand_option_or_answer", "If options exist, return the option letter; otherwise return the answer.", role="structured_output", components=["format", "concise"], reference=PROMPTBENCH, source_note="Option-letter or answer candidate."),
    record("cand_no_markdown_answer", "Return the answer plainly, without markdown.", role="structured_output", components=["format", "short"], reference=PROMPT_REPORT, source_note="Plain formatting candidate."),
    record("cand_careful_final", "Read carefully, then give only the answer.", role="careful_brevity", components=["careful", "concise", "short"], reference=PROMPT_REPORT, source_note="Careful plus answer-only candidate."),
    record("cand_constraints_final", "Check the constraints, then give only the answer.", role="careful_brevity", components=["careful", "concise", "short"], reference=PROMPT_REPORT, source_note="Constraint-attention answer-only candidate."),
    record("cand_details_short", "Notice the details and answer briefly.", role="careful_brevity", components=["careful", "short"], reference=PROMPT_REPORT, source_note="Detail-attention brevity candidate."),
    record("cand_precise_no_assumptions", "Avoid assumptions; answer briefly.", role="careful_brevity", components=["careful", "short"], reference=PROMPT_REPORT, source_note="Assumption-avoidance concise candidate."),
    record("cand_interpret_short", "Interpret the question carefully and answer briefly.", role="careful_brevity", components=["careful", "strategy", "short"], reference=PROMPT_REPORT, source_note="Interpretation plus brevity candidate."),
    record("cand_question_type_answer", "Identify the question type, then answer only.", role="careful_brevity", components=["careful", "strategy", "concise"], reference=PROMPT_REPORT, source_note="Question-type identification candidate."),
    record("cand_check_final", "Double-check internally, then give only the answer.", role="check_brevity", components=["check", "concise", "short"], reference=SPRIG, source_note="Verification plus answer-only candidate."),
    record("cand_sanity_final", "Do a quick sanity check, then answer only.", role="check_brevity", components=["check", "concise", "short"], reference=SPRIG, source_note="Sanity-check answer-only candidate."),
    record("cand_verify_constraints_final", "Verify the constraints internally. Return only the answer.", role="check_brevity", components=["check", "careful", "concise"], reference=SPRIG, source_note="Constraint verification answer-only candidate."),
    record("cand_reconsider_final", "Reconsider once if uncertain, then give only the answer.", role="check_brevity", components=["check", "concise"], reference=SPRIG, source_note="Uncertainty check answer-only candidate."),
    record("cand_expert_short_answer", "As an expert, give only the answer.", role="expert_brevity", components=["expert", "concise", "short"], reference=EXPERT_PROMPTING, source_note="Expert plus answer-only candidate."),
    record("cand_expert_exact", "As a domain expert, give the exact answer briefly.", role="expert_brevity", components=["expert", "short"], reference=EXPERT_PROMPTING, source_note="Expert exact concise candidate."),
    record("cand_meticulous_expert_short", "As a meticulous expert, answer briefly and accurately.", role="expert_brevity", components=["expert", "careful", "short"], reference=EXPERT_PROMPTING, source_note="Meticulous expert brevity candidate."),
    record("cand_specialist_final", "Use specialist judgment and return only the answer.", role="expert_brevity", components=["expert", "concise"], reference=EXPERT_PROMPTING, source_note="Specialist judgment answer-only candidate."),
    record("cand_examiner_short", "Answer like an expert examiner, briefly.", role="expert_brevity", components=["expert", "role", "short"], reference=EXPERT_PROMPTING, source_note="Expert examiner brevity candidate."),
    record("cand_emotion_short", "This matters a lot. Give only the answer.", role="emotion_brevity", components=["emotion", "concise", "short"], reference=EMOTION_PROMPT, source_note="EmotionPrompt-style answer-only candidate."),
    record("cand_career_short", "This is important to my career. Answer briefly.", role="emotion_brevity", components=["emotion", "short"], reference=EMOTION_PROMPT, source_note="Career-importance brevity candidate."),
    record("cand_high_stakes_final", "The answer matters. Return only the final answer.", role="emotion_brevity", components=["emotion", "concise"], reference=EMOTION_PROMPT, source_note="High-stakes answer-only candidate."),
    record("cand_best_short", "Please give your best answer, briefly.", role="emotion_brevity", components=["emotion", "short"], reference=EMOTION_PROMPT, source_note="Best-effort brevity candidate."),
    record("cand_calm_short", "Stay calm and answer briefly.", role="emotion_brevity", components=["emotion", "short"], reference=EMOTION_PROMPT, source_note="Calm affect plus brevity candidate."),
    record("cand_brief_internal_reason", "Think briefly internally, then give only the answer.", role="soft_reason_brevity", components=["soft_reason", "concise"], reference=ZERO_SHOT_COT, source_note="Internal soft-reasoning answer-only candidate.", complexity_level="lightweight"),
    record("cand_internal_plan_final", "Make a tiny internal plan, then return only the answer.", role="soft_reason_brevity", components=["soft_reason", "strategy", "concise"], reference=PROMPT_REPORT, source_note="Internal plan answer-only candidate.", complexity_level="lightweight"),
    record("cand_compute_internal_final", "Compute internally if needed, then give only the answer.", role="soft_reason_brevity", components=["soft_reason", "concise"], reference=ZERO_SHOT_COT, source_note="Internal computation answer-only candidate.", complexity_level="lightweight"),
    record("cand_compare_options_final", "Compare options internally, then return only the answer.", role="soft_reason_brevity", components=["soft_reason", "strategy", "concise"], reference=PROMPT_REPORT, source_note="Internal option-comparison answer-only candidate.", complexity_level="lightweight"),
    record("cand_systematic_short", "Solve systematically but answer briefly.", role="soft_reason_brevity", components=["soft_reason", "strategy", "short"], reference=PROMPT_REPORT, source_note="Systematic concise candidate.", complexity_level="lightweight"),
    record("cand_structure_final", "Use the task structure and give only the answer.", role="meta_brevity", components=["meta_prompting", "strategy", "concise"], reference=META_PROMPTING, source_note="Meta-structure answer-only candidate."),
    record("cand_formal_short", "Focus on the formal structure. Answer briefly.", role="meta_brevity", components=["meta_prompting", "short"], reference=META_PROMPTING, source_note="Formal-structure brevity candidate."),
    record("cand_task_intent_final", "Infer the task intent, then return only the answer.", role="strategy_brevity", components=["strategy", "concise"], reference=PROMPT_REPORT, source_note="Task-intent answer-only candidate."),
    record("cand_strategy_short", "Choose the right strategy and answer briefly.", role="strategy_brevity", components=["strategy", "short"], reference=PROMPT_REPORT, source_note="Strategy brevity candidate."),
    record("cand_abstract_short", "Use general principles and answer briefly.", role="strategy_brevity", components=["strategy", "short"], reference=META_PROMPTING, source_note="Abstract strategy brevity candidate."),
    record("cand_relevant_facts_final", "Use relevant facts internally, then give only the answer.", role="knowledge_brevity", components=["knowledge", "concise"], reference=GENERATED_KNOWLEDGE, source_note="Relevant-facts answer-only candidate."),
    record("cand_background_short", "Recall useful background knowledge and answer briefly.", role="knowledge_brevity", components=["knowledge", "short"], reference=GENERATED_KNOWLEDGE, source_note="Background-knowledge brevity candidate."),
    record("cand_commonsense_final", "Use commonsense knowledge and return only the answer.", role="knowledge_brevity", components=["knowledge", "concise"], reference=GENERATED_KNOWLEDGE, source_note="Commonsense answer-only candidate."),
]


def parse_args():
    parser = argparse.ArgumentParser(description="Build activation-selection candidate prompts.")
    parser.add_argument("--output", default="data/prompts_activation_candidates.jsonl")
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
    print(f"[build_activation_candidate_prompt_pool] wrote {len(PROMPTS)} prompts to {output_path}")


if __name__ == "__main__":
    main()
