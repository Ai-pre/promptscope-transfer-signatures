# Next Experiments

## Goal

Move the project from a pilot result to a more defensible claim:

1. Reproducibility: rerun the same pipeline on newer, more canonical accessible checkpoints.
2. Interpretability: separate mixed reference directions into component-level prompt directions.
3. Practicality: test whether activation information can guide prompt design better than raw cosine selection.

## Experiment 1: Newer Model Reproducibility

The previous pilot used accessible substitutes such as Hermes and FuseChat-Gemma. For a stronger follow-up, rerun the reference prompt experiment and activation-informed prompt design on newer Unsloth checkpoints.

Recommended configs:

- `configs/config.unsloth_qwen3_4b_general_mixed_clean.yaml`
- `configs/config.unsloth_qwen3_4b_principle_boundary.yaml`
- `configs/config.unsloth_llama4_scout_general_mixed_clean.yaml`
- `configs/config.unsloth_llama4_scout_principle_boundary.yaml`
- `configs/config.unsloth_gemma4_e2b_general_mixed_clean.yaml`
- `configs/config.unsloth_gemma4_e2b_principle_boundary.yaml`

Optional larger Qwen3.6 run if the server can handle it:

- `configs/config.unsloth_qwen36_27b_general_mixed_clean.yaml`
- `configs/config.unsloth_qwen36_27b_principle_boundary.yaml`

For each model family, run:

```bash
python scripts/run_eval.py --config <general_config>
python scripts/extract_activation.py --config <general_config>
python scripts/run_analysis.py --config <general_config>

python scripts/run_eval.py --config <principle_config>
python scripts/extract_activation.py --config <principle_config>
python scripts/run_analysis.py --config <principle_config>

python scripts/run_principle_analysis.py \
  --reference-config <general_config> \
  --config <principle_config>

python scripts/run_activation_selection.py \
  --reference-config <general_config> \
  --candidate-config <principle_config>
```

Success signal:

- Activation-based top-5 beats random top-5.
- The strongest slice again appears near a user-input boundary or another interpretable layer/position.
- Lightweight boundary-setting prompts remain competitive with or better than heavy scaffold prompts.

## Experiment 2: Component Direction Decomposition

The current reference centroid is a mixture direction. It averages the activation signatures of strong paper-backed prompts, but those prompts contain many components at once: concise wording, answer format, reasoning cue, persona, verbosity, and carefulness.

Therefore, a high cosine with the reference centroid is hard to interpret. It does not tell us whether a prompt is close because of `concise`, `format`, `careful`, or some mixture of all of them.

Next, define component directions using controlled prompt pairs:

```text
concise_direction = delta_h(concise_prompt) - delta_h(plain_prompt)
format_direction = delta_h(format_prompt) - delta_h(plain_prompt)
careful_direction = delta_h(careful_prompt) - delta_h(plain_prompt)
check_direction = delta_h(check_prompt) - delta_h(plain_prompt)
soft_reason_direction = delta_h(soft_reason_prompt) - delta_h(plain_prompt)
```

Then compare high-performing prompts against these component directions:

```text
cosine(prompt_delta_h, concise_direction)
cosine(prompt_delta_h, format_direction)
cosine(prompt_delta_h, careful_direction)
```

Key question:

- Is the good-prompt activation direction mostly explained by concise wording, minimal answer formatting, careful reading, or a combination?

Success signal:

- High-performing prompts align more with lightweight component directions than heavy scaffold directions.
- Component-direction similarity is more interpretable than raw reference-centroid cosine.
- The same component direction appears useful across at least two model families.

## Experiment 3: Better Activation-Guided Prompt Design

Raw reference-centroid cosine was not reliable enough as a standalone prompt selection score.

Instead of using one mixed centroid, try more structured scoring:

- supervised probe score from activation signatures
- normalized component directions
- task-conditioned centroids
- component direction decomposition

Candidate scoring examples:

```text
score(prompt) =
  cosine(prompt_delta_h, concise_direction)
  + cosine(prompt_delta_h, format_direction)
  + cosine(prompt_delta_h, careful_direction)
```

or:

```text
score(prompt) = ridge_prediction(prompt_delta_h)
```

Success signal:

- Structured activation-guided scoring beats random selection.
- Ideally it approaches or beats seen-task selection.
- The score is interpretable in terms of prompt components, not just a black-box activation centroid.
