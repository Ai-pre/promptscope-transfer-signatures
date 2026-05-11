# PromptScope: Transferable System Prompt Signatures

This repository implements an end-to-end pilot pipeline for testing whether transferable system prompts produce shared activation signatures and whether those signatures predict unseen-task transfer.

## Current finding

Across `unsloth/Qwen3-4B-Instruct-2507`, `unsloth/gemma-4-E2B-it`, and `unsloth/Meta-Llama-3.1-8B-Instruct`, activation signatures contain transfer-related diagnostic signal. The most robust claim is not that activation signatures are a complete prompt selector or causal steering vector, but that they help diagnose which generation mode a system prompt induces.

The final interpretation shifted after length-control and latent-trajectory analyses. Early controlled prompt results suggested lightweight boundary-setting cues such as `concise`, `careful`, and `format`; however, the more stable cross-model principle is now better described as **minimal answer mode / verbosity control**. In practice, short or answer-only prompts tend to avoid long, verbose generation trajectories. Explicit `FINAL ANSWER` formatting helps some models, especially Llama, but is not universal across Qwen and Gemma.

Raw activation-centroid similarity was not reliable enough as a standalone prompt-selection score. Single-layer/single-token activation patching also did not reproduce the prompt behavior, so the strongest claim is diagnostic rather than causal. See [Final Lab Meeting Report](docs/final_lab_meeting_report_ko.md) for the current full write-up.

## Prompt component design and references

The controlled prompt components are not intended to be arbitrary wording choices. They are derived from recurring categories in prompt-engineering surveys and representative prompting methods, then simplified into minimal system-prompt interventions.

The main taxonomy references are:

- [The Prompt Report: A Systematic Survey of Prompt Engineering Techniques](https://arxiv.org/abs/2406.06608)
- [A Systematic Survey of Prompt Engineering in Large Language Models: Techniques and Applications](https://arxiv.org/abs/2402.07927)

Component mapping:

| Component family | Local labels | Motivation / references |
| --- | --- | --- |
| Brevity / answer-only constraints | `short`, `concise` | Answer-only and direct-answer prompting patterns in benchmark prompts and prompt-engineering taxonomies |
| Structured output constraints | `format` | Final-answer or structured-output contracts used for answer extraction |
| Reasoning elicitation | `soft_reason`, `hard_reason` controls | [Chain-of-Thought](https://arxiv.org/abs/2201.11903), [Zero-shot-CoT](https://arxiv.org/abs/2205.11916) |
| Problem decomposition | least-to-most reference prompts | [Least-to-Most Prompting](https://arxiv.org/abs/2205.10625) |
| Knowledge elicitation | generated-knowledge reference prompts | [Generated Knowledge Prompting](https://arxiv.org/abs/2110.08387) |
| Verification / self-check | `check` | [Self-Refine](https://arxiv.org/abs/2303.17651), [Self-Consistency](https://arxiv.org/abs/2203.11171) |
| Role / persona prompting | expert and meta-expert reference prompts | [ExpertPrompting](https://arxiv.org/abs/2305.14688) |
| Emotional stimulus prompting | emotion reference prompts | [EmotionPrompt](https://arxiv.org/abs/2307.11760) |
| Prompt optimization / meta prompting | PromptWizard and meta-prompting reference prompts | [PromptWizard](https://arxiv.org/abs/2405.18369), [Meta Prompting](https://arxiv.org/abs/2311.11482) |

The controlled prompt files split this into three levels:

- `data/prompts_principle_boundary.jsonl` tests `concise`, `careful`, `format`, `check`, and `soft_reason` as minimal prompt components. These prompts are ref-aligned: for example, `concise` uses the BBH answer-only wording, `format` uses a PromptBench-style answer-format suffix normalized to `FINAL ANSWER`, `check` uses a PromptBench/EmotionPrompt recheck cue, and `soft_reason` uses the Zero-shot-CoT cue.
- `data/prompts_length_control_boundary.jsonl` adds `short` and `verbose` controls to disentangle answer-boundary effects from output-length and verbosity effects. These controls are also ref-aligned: minimal-output prompts use BBH answer-only wording, format controls use PromptBench answer-format wording, and verbose controls use PromptWizard-style reasoning-followed-by-answer wording.
- `data/prompts_taxonomy_controls.jsonl` adds one representative controlled prompt for broader taxonomy families: expert persona, emotional stimulus, generated knowledge, least-to-most decomposition, zero-shot CoT, meta-prompting, and PromptWizard-like task-aware prompting.

The taxonomy-control pool is intentionally not an exhaustive combinatorial search. It is a small representative control set for checking whether the final minimal-answer-mode finding survives comparison against larger prompting families.

Source-derived taxonomy controls:

| Prompt id | Controlled family | Source-grounded cue used locally |
| --- | --- | --- |
| `taxonomy_answer_only` | Brevity / direct answer | Minimal answer-only control based on BBH: `Answer the question. Give only the final answer.` |
| `taxonomy_structured_output` | Structured output | PromptBench-style final-answer contract normalized to: `FINAL ANSWER: <answer>` |
| `taxonomy_zero_shot_cot` | Zero-shot chain-of-thought | Canonical Zero-shot-CoT cue from Kojima et al.: `Let's think step by step.` |
| `taxonomy_generated_knowledge` | Generated knowledge prompting | Knowledge-first cue from Liu et al.: `Generate some knowledge about the input, then answer the question.` |
| `taxonomy_least_to_most` | Problem decomposition | Least-to-most style cue: break the task into simpler subproblems before answering |
| `taxonomy_expert_persona` | Expert / role prompting | ExpertPrompting-style role cue: answer as a distinguished domain expert |
| `taxonomy_emotion` | Emotional stimulus prompting | EmotionPrompt-style cue: `This is important to my career.` |
| `taxonomy_meta_prompting` | Meta prompting | Meta-prompting cue: focus on the formal task structure rather than content-specific examples |
| `taxonomy_promptwizard_like` | Prompt optimization | PromptWizard-style cue: task-aware optimized instruction with expert perspective and concise strategy |

## What is included

- `scripts/run_eval.py`: runs prompt-level generation and task evaluation
- `scripts/extract_activation.py`: extracts hidden states and computes `delta_h = h(prompt) - h(base_prompt)`
- `scripts/run_analysis.py`: computes similarity, stability, and transfer prediction baselines
- `scripts/run_length_confound_analysis.py`: checks whether component effects remain after controlling for generated length, prompt length, and final-answer marker rate
- `scripts/build_paper_backed_prompt_pool.py`: builds a paper-backed prompt pool from official repositories
- `scripts/build_prompt_subsets.py`: builds filtered prompt pools for mixed vs. strict generalization runs
- `data/prompts_paper_backed.jsonl`: default source-backed prompt pool with provenance metadata
- `data/prompts.jsonl`: legacy starter pool kept for reference only
- `data/datasets/*.json`: small schema examples for each task

## Dataset schema

Each dataset file should be a JSON list with records like:

```json
{
  "id": "gsm8k-0",
  "input": "Question text here",
  "label": "42"
}
```

Optional fields:

- `valid_choices`: valid answer labels for multiple choice tasks such as `["A", "B", "C", "D", "E"]`

## Prompt schema

Each prompt record is stored as JSONL:

```json
{
  "id": "promptwizard_gsm8k_seed",
  "group_id": "promptwizard_gsm8k_seed",
  "variant": "original",
  "source": "promptwizard",
  "source_title": "PromptWizard GSM8K seed config",
  "source_url": "https://raw.githubusercontent.com/...",
  "paper_title": "PromptWizard: Task-Aware Prompt Optimization Framework",
  "paper_url": "https://arxiv.org/abs/2405.18369",
  "optimized_for_tasks": ["gsm8k"],
  "prompt_role": "system",
  "original_prompt_role": "system",
  "task_scope": "task_specific",
  "text": "You are a mathematics expert..."
}
```

To support paraphrase stability experiments, add extra rows with the same `group_id` and different `variant` values such as `paraphrase_1`, `paraphrase_2`, `paraphrase_3`.

The default configs now point to `data/prompts_paper_backed.jsonl`, which is built from official repositories and keeps provenance fields so you can trace each prompt back to a paper/repo source.

## Run

```bash
python scripts/build_paper_backed_prompt_pool.py
python scripts/build_prompt_subsets.py
python scripts/run_eval.py
python scripts/extract_activation.py
python scripts/run_analysis.py
```

The evaluation step also writes:

- `eval_seen_fit_summary.parquet/json`: full prompt-variant ranking
- `eval_seen_fit_originals.parquet/json`: original-only ranking
- `eval_seen_fit_groups.parquet/json`: group-level ranking

These files make it easier to separate clean general experiments from paraphrase-heavy stability runs.

## Download real benchmark data

The repository ships with tiny placeholder files for quick smoke tests. To fetch real benchmark data from the Hugging Face Hub and convert it into the local JSON schema, run:

```bash
python -m pip install -r requirements.txt
python scripts/prepare_datasets.py
```

This downloader currently uses:

- `openai/gsm8k` `train` for `gsm8k.json`
- `tau/commonsense_qa` `validation` for `csqa.json`
- `MU-NLPC/Calc-svamp` `test` for `svamp.json`
- `google/boolq` `validation` for `boolq.json`

If you only want a quick pilot subset:

```bash
python scripts/prepare_datasets.py --max-samples 100
```

## Server-friendly smoke test

If your server cannot use the default 7B model with CUDA, use the smaller CPU-oriented config:

```bash
python scripts/run_eval.py --config configs/config.cpu_smoke.yaml
python scripts/extract_activation.py --config configs/config.cpu_smoke.yaml
python scripts/run_analysis.py --config configs/config.cpu_smoke.yaml
```

The CPU smoke config already writes into `outputs/cpu_smoke`, limits prompt count to 2, and uses a small 1.5B model.

## Recommended GPU pilot on GPU 2

For a first meaningful pilot on a multi-GPU server, use the dedicated GPU pilot config. It writes to `outputs/gpu_pilot`, uses the 7B model in fp16, limits to 10 prompts, and caps each task at 20 samples by default.

```bash
export CUDA_VISIBLE_DEVICES=2
python scripts/run_eval.py --config configs/config.gpu_pilot.yaml
python scripts/extract_activation.py --config configs/config.gpu_pilot.yaml
python scripts/run_analysis.py --config configs/config.gpu_pilot.yaml
```

You can still override prompt or task limits at the command line:

```bash
python scripts/run_eval.py --config configs/config.gpu_pilot.yaml --limit-prompts 5 --limit-per-task 10
```

## Stability pilot with paraphrases

To open the paraphrase-stability experiment, first build a prompt pool with 3 paraphrases per original prompt:

```bash
python scripts/build_paper_backed_prompt_pool.py
python scripts/build_paraphrase_prompt_pool.py --limit-groups 10
```

This writes `data/prompts_paper_backed_paraphrase.jsonl` with 10 original prompt groups and 3 paraphrases per group.

Then run the stability-oriented pilot:

```bash
export CUDA_VISIBLE_DEVICES=2
python scripts/run_eval.py --config configs/config.gpu_stability_pilot.yaml
python scripts/extract_activation.py --config configs/config.gpu_stability_pilot.yaml
python scripts/run_analysis.py --config configs/config.gpu_stability_pilot.yaml
```

The analysis step now also writes `outputs/.../results/slice_analysis.parquet` and `slice_analysis.json`, which compare activation signal by `layer`, `position`, and `layer_position` slices.

## Scale-up run

If the small pilots finish cleanly and you want a more meaningful run with larger sample counts, use the scale-up configs.

Non-paraphrase scale-up:

```bash
export CUDA_VISIBLE_DEVICES=2
python scripts/run_eval.py --config configs/config.gpu_scaleup.yaml
python scripts/extract_activation.py --config configs/config.gpu_scaleup.yaml
python scripts/run_analysis.py --config configs/config.gpu_scaleup.yaml
```

This uses:

- all paper-backed original prompts in `data/prompts_paper_backed.jsonl` (currently 13)
- 100 samples per task
- outputs in `outputs/gpu_scaleup`

Stability scale-up:

```bash
python scripts/build_paraphrase_prompt_pool.py --limit-groups 10
export CUDA_VISIBLE_DEVICES=2
python scripts/run_eval.py --config configs/config.gpu_stability_scaleup.yaml
python scripts/extract_activation.py --config configs/config.gpu_stability_scaleup.yaml
python scripts/run_analysis.py --config configs/config.gpu_stability_scaleup.yaml
```

This uses:

- 10 prompt groups x 4 variants = 40 prompt variants
- 50 samples per task
- outputs in `outputs/gpu_stability_scaleup`

## Clean generalization subsets

To maximize interpretability, you can split the paper-backed pool into:

- `data/prompts_paper_backed_mixed_seen_aligned.jsonl`: official prompts whose source task aligns with the seen split, plus task-agnostic prompts
- `data/prompts_paper_backed_strict_system_seen_aligned.jsonl`: only genuine system prompts among the aligned subset

Recommended runs:

```bash
python scripts/build_prompt_subsets.py
export CUDA_VISIBLE_DEVICES=2

python scripts/run_eval.py --config configs/config.gpu_general_mixed.yaml
python scripts/extract_activation.py --config configs/config.gpu_general_mixed.yaml
python scripts/run_analysis.py --config configs/config.gpu_general_mixed.yaml

python scripts/run_eval.py --config configs/config.gpu_general_strict.yaml
python scripts/extract_activation.py --config configs/config.gpu_general_strict.yaml
python scripts/run_analysis.py --config configs/config.gpu_general_strict.yaml
```

Use the mixed run for the strongest signal search and the strict run as the cleaner sanity-check setting.
