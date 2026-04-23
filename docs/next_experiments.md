# Next Experiments

## Goal

Move the project from a strong pilot to a paper-grade claim by adding one test for each missing axis:

1. Reproducibility: check whether the activation/principle story survives on a second model.
2. Causality: isolate lightweight boundary-setting components with minimal-pair probes.
3. Practicality: test whether activation similarity can actually select good prompts.

## Experiment 1: Cross-Model Reproducibility

Run the clean paper-backed reference set and the boundary probe set on `microsoft/Phi-3.5-mini-instruct`.

Outputs:

- `outputs/phi35_general_mixed_clean/results/analysis_summary.json`
- `outputs/phi35_general_mixed_clean/results/slice_analysis.json`
- `outputs/phi35_principle_boundary/principle_results/principle_summary.json`
- `outputs/phi35_principle_boundary/principle_results/principle_component_effects.json`
- `outputs/phi35_principle_boundary/principle_results/principle_contrasts.json`

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
