# Next Experiments

## Goal

Move the project from a strong pilot to a paper-grade claim by adding one test for each missing axis:

1. Reproducibility: check whether the activation/principle story survives on a second model.
2. Causality: isolate lightweight boundary-setting components with minimal-pair probes.
3. Practicality: test whether activation similarity can actually select good prompts.

## Experiment 1: Cross-Model Reproducibility

Run the clean paper-backed reference set and the boundary probe set on two stronger instruction-tuned families:

- `meta-llama/Llama-3.1-8B-Instruct`
- `google/gemma-2-9b-it`
- If gated access is unavailable on the server, use the ungated fallback `mistralai/Mistral-7B-Instruct-v0.3`.
- If you want a closer family-level substitute for gated checkpoints, use open derivatives:
  - `NousResearch/Hermes-3-Llama-3.1-8B` for the Llama family
  - `FuseAI/FuseChat-Gemma-2-9B-SFT` for the Gemma family

Suggested order:

1. Run Gemma first because it is usually easier to access from Hugging Face.
2. Run Llama next if the server account has accepted the Meta license / token gate.

Configs:

- `configs/config.gemma2_general_mixed_clean.yaml`
- `configs/config.gemma2_principle_boundary.yaml`
- `configs/config.llama31_general_mixed_clean.yaml`
- `configs/config.llama31_principle_boundary.yaml`
- `configs/config.mistral7_general_mixed_clean.yaml`
- `configs/config.mistral7_principle_boundary.yaml`
- `configs/config.hermes3_general_mixed_clean.yaml`
- `configs/config.hermes3_principle_boundary.yaml`
- `configs/config.fusechat_gemma_general_mixed_clean.yaml`
- `configs/config.fusechat_gemma_principle_boundary.yaml`

Outputs:

- `outputs/gemma2_general_mixed_clean/results/analysis_summary.json`
- `outputs/gemma2_general_mixed_clean/results/slice_analysis.json`
- `outputs/gemma2_principle_boundary/principle_results/principle_summary.json`
- `outputs/gemma2_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/gemma2_principle_boundary/principle_results/principle_contrasts.json`
- `outputs/llama31_general_mixed_clean/results/analysis_summary.json`
- `outputs/llama31_general_mixed_clean/results/slice_analysis.json`
- `outputs/llama31_principle_boundary/principle_results/principle_summary.json`
- `outputs/llama31_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/llama31_principle_boundary/principle_results/principle_contrasts.json`
- `outputs/mistral7_general_mixed_clean/results/analysis_summary.json`
- `outputs/mistral7_general_mixed_clean/results/slice_analysis.json`
- `outputs/mistral7_principle_boundary/principle_results/principle_summary.json`
- `outputs/mistral7_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/mistral7_principle_boundary/principle_results/principle_contrasts.json`
- `outputs/hermes3_general_mixed_clean/results/analysis_summary.json`
- `outputs/hermes3_general_mixed_clean/results/slice_analysis.json`
- `outputs/hermes3_principle_boundary/principle_results/principle_summary.json`
- `outputs/hermes3_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/hermes3_principle_boundary/principle_results/principle_contrasts.json`
- `outputs/fusechat_gemma_general_mixed_clean/results/analysis_summary.json`
- `outputs/fusechat_gemma_general_mixed_clean/results/slice_analysis.json`
- `outputs/fusechat_gemma_principle_boundary/principle_results/principle_summary.json`
- `outputs/fusechat_gemma_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/fusechat_gemma_principle_boundary/principle_results/principle_contrasts.json`

Success signal:

- Activation predicts unseen transfer above random.
- Best slice is again concentrated around user-input boundary positions.
- Lightweight boundary-setting beats heavy or intrusive controls.

## Experiment 2: Minimal-Pair Boundary Probes

Use `configs/config.gpu_principle_boundary.yaml`.

This isolates:

- `concise`
- `careful`
- `check`
- `format`
- `soft_reason`
- combinations such as `concise_careful_format`

Outputs:

- `outputs/gpu_principle_boundary/results/analysis_summary.json`
- `outputs/gpu_principle_boundary/principle_results/principle_summary.json`
- `outputs/gpu_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/gpu_principle_boundary/principle_results/principle_contrasts.json`

Success signal:

- `concise` improves unseen performance.
- `careful` or `check` improves activation similarity to the reference centroid.
- Combined lightweight boundary prompts preserve or improve unseen accuracy without becoming verbose.

## Experiment 3: Activation-Based Selection

Use `scripts/run_activation_selection.py` after both the reference and candidate runs are complete.

Outputs:

- `outputs/gpu_principle_boundary/activation_selection/activation_selection_summary.json`
- `outputs/gpu_principle_boundary/activation_selection/activation_selection_group_table.json`
- `outputs/gpu_principle_boundary/activation_selection/activation_selection_report.md`

Success signal:

- Ranking candidates by reference-centroid cosine beats random selection.
- Ideally, activation selection approaches or beats seen-accuracy selection.
