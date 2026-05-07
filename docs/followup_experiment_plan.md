# Follow-Up Experiments After Literature Review

This plan turns the literature-review suggestions into runnable experiments.

## Experiment 1. Boundary-Localized Component Patching

Question: does a component direction such as `concise - plain` causally change unseen-task behavior?

Script:

```bash
python scripts/run_component_patching.py \
  --config configs/config.unsloth_qwen3_4b_principle_expanded.yaml \
  --positive-prompt-id principle3_concise \
  --negative-prompt-id principle3_plain \
  --target-prompt-id principle3_plain \
  --remove-from-prompt-id principle3_concise \
  --direction-position first_user_token \
  --tasks svamp boolq bbeh_mini gpqa \
  --limit-per-task 25 \
  --alphas 1.0
```

Run the same command for:

```bash
configs/config.unsloth_gemma4_e2b_principle_expanded.yaml
configs/config.unsloth_llama31_8b_principle_expanded.yaml
```

Main outputs:

```text
outputs/<run>/component_patching_results/component_patching_summary.json
outputs/<run>/component_patching_results/component_patching_report.md
```

Interpretation:

- Supportive evidence: `activation_patch` improves over `target_baseline` more than matched-norm random controls.
- Stronger evidence: `direction_removal` lowers the donor prompt relative to `donor_baseline`.

## Experiment 2. Position Localization

Question: is the effect concentrated at the system/user boundary?

The same patching script tests multiple positions:

```text
system_last_token
first_user_token
user_middle_token
user_last_token
all_user_tokens
first_generation_token
```

Interpretation:

- Boundary story is strong if `system_last_token` or `first_user_token` works better than user-middle, user-last, and first-generation-token patching.
- If every position works similarly, the effect is more like generic steering than boundary-state reconfiguration.

## Experiment 3. Cross-Model Direction Alignment

Question: is the concise principle represented similarly across model families?

Script:

```bash
python scripts/run_cross_model_direction_alignment.py \
  --configs \
    configs/config.unsloth_qwen3_4b_principle_expanded.yaml \
    configs/config.unsloth_gemma4_e2b_principle_expanded.yaml \
    configs/config.unsloth_llama31_8b_principle_expanded.yaml
```

Main outputs:

```text
outputs/cross_model_direction_alignment/cross_model_alignment_summary.json
outputs/cross_model_direction_alignment/cross_model_alignment_report.md
```

Interpretation:

- Because hidden dimensions differ across models, direct vector cosine is not always meaningful.
- The script therefore reports feature CKA and prompt-wise component-score correlation.
- Safe claim if successful: the principle has family-specific implementations but partially aligned behavior-level geometry.

## Experiment 4. Length-Control Disentanglement

Question: is `concise` useful because it is short, or because it sets an answer boundary?

Build prompts:

```bash
python scripts/build_length_control_prompt_pool.py
```

Run evaluation for each model:

```bash
python scripts/run_eval.py --config configs/config.unsloth_qwen3_4b_length_control_expanded.yaml --limit-per-task 50
python scripts/run_length_control_analysis.py --config configs/config.unsloth_qwen3_4b_length_control_expanded.yaml
```

Repeat for:

```bash
configs/config.unsloth_gemma4_e2b_length_control_expanded.yaml
configs/config.unsloth_llama31_8b_length_control_expanded.yaml
```

Main outputs:

```text
outputs/<run>/length_control_results/length_control_summary.json
outputs/<run>/length_control_results/length_control_report.md
```

Interpretation:

- If `boundary_only` beats `length_only`, the principle should be framed as answer-boundary control rather than mere brevity.
- If `length_only` wins, the previous concise result may be mostly decoding-style or output-length control.

## Experiment 5. Latent Reasoning Trajectory Analysis

Question: does a concise/boundary prompt change early hidden-state dynamics after the user question?

Script:

```bash
python scripts/run_latent_trajectory_analysis.py \
  --config configs/config.unsloth_qwen3_4b_principle_expanded.yaml \
  --prompt-ids principle3_plain principle3_concise principle3_concise_format principle3_concise_careful_format \
  --tasks svamp boolq bbeh_mini gpqa \
  --limit-per-task 20 \
  --generated-token-count 12
```

Main outputs:

```text
outputs/<run>/latent_trajectory_results/latent_trajectory_summary.json
outputs/<run>/latent_trajectory_results/latent_trajectory_report.md
```

Interpretation:

- If concise/boundary prompts show shorter or more direct early trajectories while improving accuracy, that supports the boundary-state reconfiguration story.
- If only output length changes and trajectory metrics do not, the mechanism is likely mostly decoding style.

## Robustness Add-On

This is not a separate conceptual experiment, but it is important for defensibility.

```bash
python scripts/run_robustness_analysis.py \
  --config configs/config.unsloth_qwen3_4b_general_expanded.yaml \
  --bootstrap-trials 1000 \
  --permutation-trials 1000 \
  --random-direction-trials 1000
```

Run this on all three general expanded configs.

Main outputs:

```text
outputs/<run>/robustness_results/robustness_summary.json
outputs/<run>/robustness_results/robustness_report.md
```

Interpretation:

- Use bootstrap CI and permutation p-value to avoid overclaiming from a single ridge R2/top-k number.
- Use leave-one-seen-task-out to check whether one seen dataset dominates the activation signal.
