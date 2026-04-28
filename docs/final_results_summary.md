# Final Results Summary

## Core Claim

Transferable prompting is better explained by lightweight boundary-setting cues than by heavy expert/reasoning scaffolds.

Across Qwen2.5-7B, Hermes-3-Llama-3.1-8B, and FuseChat-Gemma-2-9B-SFT, activation features consistently contained transfer-related signal. The strongest signals repeatedly involved the user-input boundary, especially `first_user_token`, although the exact layer varied by model. Principle probes showed that the most reliable prompt family was built from concise, careful, and minimal answer-format cues. Direct activation-centroid selection was not reliable.

## Model-Level Reference Results

| Model | Activation R2 | Activation top-k unseen | Seen top-k unseen | Random top-k unseen | Main localization |
|---|---:|---:|---:|---:|---|
| Qwen2.5-7B-Instruct | 0.438 | 0.585 | 0.591 | 0.480 | Layer 28, `first_user_token` |
| Hermes-3-Llama-3.1-8B | 0.405 | 0.789 | 0.802 | 0.644 | `first_user_token`, early/mid layers |
| FuseChat-Gemma-2-9B-SFT | 0.256 | 0.640 | 0.600 | 0.543 | Layer 20/42, user-boundary slices |

Interpretation:

Activation is a real transfer-related signal across model families. It often beats random selection and sometimes beats seen-task selection, but it is not uniformly stronger than the seen-task baseline.

## Principle Probe Results

| Model | Best unseen probe | Best activation-similarity probe | Main interpretation |
|---|---|---|---|
| Qwen2.5-7B-Instruct | `principle3_concise_careful_format` | `principle3_concise_careful_format` | Performance and activation alignment converge on the same lightweight boundary prompt. |
| Hermes-3-Llama-3.1-8B | `principle3_careful_format_check` | `principle3_soft_reason` | Boundary prompts remain strong, but the exact best composition shifts by model. |
| FuseChat-Gemma-2-9B-SFT | `principle3_concise_careful_format` | `principle3_soft_reason_format` | The performance winner again matches the concise/careful/format family. |

Interpretation:

The exact optimal prompt is not universal, but the family is stable: short boundary-setting cues are more reliable than strong expert personas, hard step-by-step reasoning, verbose scaffolds, or multi-agent instructions.

## Component-Level Pattern

### Recurring Positive Signals

`concise` was the clearest performance driver.

Qwen: `+0.106` unseen accuracy  
FuseChat-Gemma: `+0.080` unseen accuracy  
Hermes: slightly negative, suggesting the exact cue is model-sensitive

`format` helped activation alignment and was often performance-neutral or positive.

Qwen: `+0.060` reference cosine  
FuseChat-Gemma: `+0.102` reference cosine  
Hermes: close to neutral

`careful` often helped activation alignment more than raw performance.

Qwen: `+0.071` reference cosine  
FuseChat-Gemma: `+0.114` reference cosine  
Hermes: not aligned with the reference centroid, showing model-level variation

### Recurring Negative Signals

Earlier principle probes showed that strong expert personas, hard reasoning instructions, verbose scaffolds, and multi-agent structures generally hurt transfer or failed to explain the reference activation pattern.

## Activation Selection

Activation-centroid selection did not become a reliable prompt-selection rule.

| Model | Activation-cosine top-k unseen | Seen top-k unseen | Random top-k unseen | Spearman with unseen |
|---|---:|---:|---:|---:|
| Qwen2.5-7B-Instruct | 0.648 | 0.757 | 0.684 | -0.355 |
| Hermes-3-Llama-3.1-8B | 0.776 | 0.810 | 0.786 | -0.256 |
| FuseChat-Gemma-2-9B-SFT | 0.700 | 0.766 | 0.733 | -0.476 |

Interpretation:

Activation signatures are useful for mechanism analysis, but raw centroid cosine is not a stable standalone selection criterion.

## Final Takeaway

The strongest supported claim is:

Transferable prompts do not work mainly by forcing the model into heavier expert or reasoning modes. They work better when they lightly set the boundary for reading the user input and committing to an answer. The most robust family is concise/careful/format-style prompting, but the exact best composition is model-dependent.

