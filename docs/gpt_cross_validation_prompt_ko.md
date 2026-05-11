# GPT 교차검증용 보고서 프롬프트

아래 내용을 다른 GPT/LLM에게 그대로 붙여넣고, 연구 결론이 과장되었는지 교차검증해 주세요.

---

## 1. 당신의 역할

너는 LLM prompting / mechanistic interpretability / empirical ML 논문을 리뷰하는 까다로운 reviewer다.

내 목표는 칭찬이 아니라 **결론의 타당성 검증**이다. 다음을 냉정하게 평가해 달라.

- 주장과 실험 증거가 제대로 대응되는가?
- activation signature라는 용어와 해석이 과장되지 않았는가?
- seen/unseen transfer를 general capability proxy로 쓰는 것이 타당한가?
- controlled prompt component가 reference-backed taxonomy에 의해 정당화되는가?
- length/output parsing confound가 결론을 설명할 가능성은 없는가?
- causal claim을 어디까지 할 수 있고, 어디서 멈춰야 하는가?
- 학부생 연구 / workshop paper 수준으로 충분한가?

가능하면 다음 형식으로 답해 달라.

1. 가장 안전한 핵심 claim
2. 과장된 claim 또는 빼야 할 표현
3. 실험 설계상 가장 큰 약점
4. 현재 결과로 지지되는 것과 지지되지 않는 것
5. 논문/보고서에 넣을 수 있는 최종 thesis 문장
6. 추가로 꼭 필요한 실험 3개 이하
7. 전체 연구 수준 평가: 학부 연구 / workshop / main conference 중 어디쯤인지

---

## 2. 연구 한 줄 요약

이 프로젝트는 system prompt가 unseen task로 transfer되는 정도를 general capability의 proxy로 보고, 좋은 prompt가 만드는 **activation signature**가 transferable prompting behavior를 진단할 수 있는지 분석한다.

현재 가장 안전한 결론은 다음이다.

> transferable prompt behavior는 모델 내부 activation signature에 어느 정도 diagnostic signal을 남기지만, 현재 실험만으로 그 signature를 causal steering vector라고 보기는 어렵다. Controlled prompt 실험과 length-control 실험을 종합하면, 안정적으로 보이는 prompting principle은 강한 reasoning scaffold나 expert persona보다 **minimal answer mode / verbosity control**에 가깝다.

---

## 3. 핵심 용어

이 프로젝트에서 activation 관련 표현은 **activation signature**로 통일한다.

```text
activation signature = delta_h = h(prompt) - h(base_prompt)
base_prompt = "You are a helpful assistant."
```

즉 prompt 자체의 hidden state가 아니라, neutral base prompt 대비 해당 prompt가 모델 내부 residual/hidden representation을 얼마나 바꾸는지 보는 차이 벡터다.

Ridge regression에서:

```text
X = prompt별 activation signature feature
y = prompt별 seen/unseen transfer-related score
```

여기서 y는 주로 prompt-level accuracy 또는 transfer score다. Ridge R2가 높다는 것은 activation signature가 prompt 성능 차이를 설명하는 predictive signal을 가진다는 뜻이지, 곧바로 causal mechanism이라는 뜻은 아니다.

---

## 4. 모델과 task split

사용 모델:

| Family | Model |
| --- | --- |
| Qwen | `unsloth/Qwen3-4B-Instruct-2507` |
| Gemma | `unsloth/gemma-4-E2B-it` |
| Llama | `unsloth/Meta-Llama-3.1-8B-Instruct` |

Task split:

| Split | Tasks | Role |
| --- | --- | --- |
| Seen | GSM8K, CSQA, BBH, MMLU-Pro | prompt selection / activation probe fitting |
| Unseen | SVAMP, BoolQ, BBEH-mini, GPQA | cross-task generalization evaluation |

seen/unseen은 prompt 출처가 아니라 task 역할이다. seen에서 좋은 prompt가 unseen에서도 좋은지, 또는 seen에서 얻은 activation signature가 unseen 성능을 예측하는지 본다.

---

## 5. Prompt pool과 reference grounding

### 5.1 Stage 1: paper-backed prompt pool

첫 실험은 이미 paper-backed prompt pool로 진행되었다. 여기에 포함된 범주는 다음과 같다.

- PromptBench CoT / Zero-shot-CoT
- PromptBench baseline answer-format prompts
- PromptWizard GSM8K seed prompt
- Generated Knowledge prompting
- Least-to-Most prompting
- ExpertPrompting
- EmotionPrompt
- BBH answer-only / BBH zero-shot-CoT
- MMLU-Pro direct-letter / knowledge-CoT prompt
- Meta-prompting variants

이 단계는 “seen task에서 성능 좋은 paper-backed prompt가 unseen에서도 좋은가?”와 “activation signature로 unseen transfer를 예측할 수 있는가?”를 본다.

### 5.2 Stage 2: controlled component prompt pool

초기에는 직접 설계한 controlled prompts로 `concise`, `careful`, `format`, `check`, `soft_reason`을 분해했다.

하지만 reviewer가 “component가 임의적이지 않은가?”라고 물을 수 있기 때문에, 현재는 Prompt Report와 대표 prompting papers에 맞춰 ref-aligned wording으로 다시 정렬했다.

Reference taxonomy:

- The Prompt Report: `https://arxiv.org/abs/2406.06608`
- Prompt engineering survey: `https://arxiv.org/abs/2402.07927`
- BBH answer-only: `https://arxiv.org/abs/2210.09261`
- PromptBench: `https://arxiv.org/abs/2312.07910`
- Zero-shot-CoT: `https://arxiv.org/abs/2205.11916`
- EmotionPrompt: `https://arxiv.org/abs/2307.11760`
- PromptWizard: `https://arxiv.org/abs/2405.18369`
- Generated Knowledge: `https://arxiv.org/abs/2110.08387`
- Least-to-Most: `https://arxiv.org/abs/2205.10625`
- ExpertPrompting: `https://arxiv.org/abs/2305.14688`
- Meta Prompting: `https://arxiv.org/abs/2311.11482`

Prompt component mapping:

| Local component | Reference-backed interpretation |
| --- | --- |
| `concise`, `short` | answer-only / direct-answer / brevity |
| `format` | structured output / answer-format constraint |
| `soft_reason`, `hard_reason` | reasoning elicitation / CoT / Zero-shot-CoT |
| `check` | verification / self-check / recheck cue |
| `careful` | instruction quality / task attention |
| `expert` | role / persona prompting |
| `emotion` | emotional stimulus prompting |
| `knowledge` | generated knowledge prompting |
| `decomposition` | least-to-most / decomposition prompting |
| `meta_prompting`, `prompt_optimization` | meta-prompting / prompt optimization |

Important current status:

- Old controlled results were produced with earlier designed wording.
- The prompt files have now been ref-aligned.
- `principle_expanded` is being rerun with `limit-per-task=100`.
- Therefore, old controlled prompt results should be interpreted as pilot evidence, not final ref-aligned evidence.
- Stage 1 paper-backed seen/unseen selection does not need rerun because it already used paper-backed prompts.

---

## 6. 실험 흐름

### Step 1. Paper-backed seen/unseen transfer experiment

Goal:

- paper-backed prompts를 seen/unseen tasks에 모두 적용
- seen에서 좋은 prompt가 unseen에서도 좋은지 확인
- activation signature가 unseen transfer를 예측하는지 확인

Metrics:

- `activation_ridge_r2`
- `activation_logistic_accuracy`
- `activation_top_k_unseen_accuracy`
- `seen_accuracy_top_k_unseen_accuracy`
- `random_top_k_unseen_accuracy`

Interpretation:

- seen accuracy만으로 transfer prompt를 고르는 것보다 activation signature가 추가 diagnostic signal을 줄 수 있는지 확인한다.

### Step 2. Controlled principle decomposition

Goal:

- paper-backed prompt에서 섞여 있던 요소를 최소 component로 나눔
- `concise`, `careful`, `format`, `check`, `soft_reason` 등이 unseen transfer에 어떤 영향을 주는지 확인

Important:

- 기존 결과는 old designed wording 기준
- 현재 ref-aligned wording으로 재실행 중

### Step 3. Structured/component activation design

Goal:

- 좋은 prompt의 activation signature와 비슷한 component direction이 실제 unseen 성능 좋은 prompt를 고르는지 확인

Finding:

- structured/component score는 일부 signal이 있으나 raw centroid similarity만으로는 universal selector라고 보기 어렵다.

### Step 4. Boundary-localized activation patching

Goal:

- `concise_direction = activation(concise) - activation(plain)`이 실제 behavior를 causal하게 매개하는지 확인
- neutral/plain run에 concise direction을 더하고, concise run에서는 그 방향을 제거
- random orthogonal direction / matched-norm noise와 비교

Finding:

- hook diagnostics상 patch는 적용되었지만 behavior 변화가 거의 없었다.
- 따라서 현재 방식에서는 activation signature를 causal steering vector라고 주장하기 어렵다.
- 안전한 해석은 diagnostic readout이다.

### Step 5. Length-control disentanglement

Goal:

- concise 효과가 “답변 경계 명확화” 때문인지, 단순히 “짧게 답하기” 때문인지 분리

Compared prompts:

- length-only
- concise-only
- boundary-only
- concise-boundary
- verbose-boundary
- verbose-no-boundary

Finding:

- Qwen/Gemma에서는 short/minimal output 계열이 가장 강함
- Llama에서는 `concise_boundary`가 특히 강함
- verbose 계열은 대체로 성능을 낮춤
- 따라서 “boundary control”보다 더 안전한 최종 principle은 “minimal answer mode / verbosity control”

### Step 6. Cross-model alignment

Goal:

- family마다 activation signature가 완전히 같은 벡터는 아니더라도, shared low-dimensional behavior/subspace가 있는지 확인

Finding:

- feature CKA는 비교적 높음
- component score correlation은 특히 `concise`, `format`에서 높게 나옴
- 하지만 direct direction cosine은 hidden dimension mismatch 때문에 NaN
- interpretation: behavior-level principle은 공유될 수 있으나 internal implementation은 model-specific

### Step 7. Robustness

Goal:

- prompt-pool artifact나 random baseline이 아닌지 확인

Methods:

- bootstrap CI
- permutation test
- random-direction baseline

Finding:

- activation R2는 permutation baseline 대비 유의미함
- top-k selection은 random보다 대체로 높지만 model/task별 variability가 있음

### Step 8. Latent trajectory analysis

Goal:

- concise/minimal prompt가 reasoning을 없애는지, 아니면 generation trajectory를 짧고 안정적으로 만드는지 확인

Metrics:

- trajectory path length
- endpoint displacement
- mean step norm
- curvature ratio
- output length

Finding:

- concise/short prompts는 generation trajectory length와 curvature를 줄이는 경향
- 하지만 이 분석은 output length confound가 강하므로 causal reasoning trajectory claim은 조심해야 함

---

## 7. 지금까지 주요 결과 요약

### 7.1 General paper-backed selection

Qwen:

- best slice: layer 18, `first_user_token`
- activation top-k unseen accuracy: about 0.361
- seen top-k unseen accuracy: about 0.387
- random top-k unseen accuracy: about 0.303
- activation R2 around 0.615 in robust slice

Gemma:

- best slice: position `first_user_token`
- activation top-k unseen accuracy: about 0.364
- seen top-k unseen accuracy: about 0.304
- random top-k unseen accuracy: about 0.260
- activation R2 around 0.790

Llama:

- best slice: layer 16, `first_user_token`
- activation top-k unseen accuracy: about 0.413
- seen top-k unseen accuracy: about 0.368
- random top-k unseen accuracy: about 0.345
- activation R2 around 0.738

Interpretation:

- activation signature contains transfer-related predictive signal.
- It is not always better than seen accuracy baseline, but it is generally above random and useful diagnostically.

### 7.2 Robustness

Qwen:

- observed activation R2: 0.615
- bootstrap R2 mean: 0.587
- CI: about [0.290, 0.779]
- permutation p-value: about 0.001

Gemma:

- observed activation R2: 0.790
- bootstrap R2 mean: 0.783
- CI: about [0.600, 0.925]
- permutation p-value: about 0.001

Llama:

- observed activation R2: 0.738
- bootstrap R2 mean: 0.724
- CI: about [0.531, 0.870]
- permutation p-value: about 0.001

Interpretation:

- activation signature signal is unlikely to be pure random artifact in the current setup.
- However, sample/task scale is still limited.

### 7.3 Cross-model alignment

Pairwise feature CKA:

- Qwen-Gemma: about 0.821
- Qwen-Llama: about 0.752
- Gemma-Llama: about 0.842

Component alignment:

- `concise` component score Spearman is high across pairs, roughly 0.75-0.84
- `format` component score Spearman is also high, roughly 0.90-0.95
- `soft_reason` is less stable

Interpretation:

- There may be behavior-level shared prompting principles.
- Direct vector identity across model families is not established.

### 7.4 Patching

Patching result:

- Adding `concise_direction` to plain prompt hidden states did not reliably improve unseen accuracy.
- Removing direction from concise prompt did not reliably degrade behavior.
- Random/matched-norm controls behaved similarly.

Interpretation:

- This weakens causal mediator claim.
- Activation signature should be framed as diagnostic/predictive, not causal steering.

### 7.5 Length-control

Old controlled wording results:

Qwen:

- best unseen prompt: `lc_length_only_one_sentence`
- best unseen accuracy: about 0.51
- prediction length correlation with unseen: about -0.842
- short component delta unseen: about +0.154
- verbose component delta unseen: about -0.155

Gemma:

- best unseen prompt: `lc_length_only_short`
- best unseen accuracy: about 0.48
- prediction length correlation with unseen: about -0.973
- short component delta unseen: about +0.190
- verbose component delta unseen: about -0.131

Llama:

- best unseen prompt: `lc_concise_boundary`
- best unseen accuracy: about 0.48
- prediction length correlation with unseen: about -0.738
- concise component delta unseen: about +0.089
- short component delta unseen: about +0.060
- verbose component delta unseen: about -0.165

Interpretation:

- The strongest recurring behavior is not necessarily explicit final-answer boundary.
- It is better described as minimal answer mode / verbosity control.
- Llama may benefit from explicit boundary marker more than Qwen/Gemma.

Important caveat:

- These results were from old controlled wording. Ref-aligned length-control rerun is still needed for final claim.

### 7.6 Latent trajectory

Representative pattern:

- verbose / boundary-only prompts often lead to longer generation trajectories
- concise / short prompts often reduce path length and curvature
- for BoolQ and GPQA, short/concise prompts can preserve or improve accuracy while reducing trajectory length

Interpretation:

- This supports the idea that minimal-answer prompts avoid long, wandering generation modes.
- But because generated output is shorter, path length reduction may partly be a mechanical length effect.
- Do not claim this proves reasoning trajectory reconfiguration unless further normalized analyses are done.

---

## 8. Current ref-aligned rerun plan

Because original controlled prompts were partly designed manually, they have now been replaced or aligned with reference-backed wording:

- `concise`: BBH answer-only wording, e.g. `Answer the question. Give only the final answer.`
- `format`: PromptBench answer-format suffix normalized to `FINAL ANSWER: <answer>`
- `soft_reason`: Zero-shot-CoT cue, `Let's think step by step.`
- `check`: PromptBench/EmotionPrompt recheck cue
- `verbose`: PromptWizard reasoning-followed-by-answer wording

Rerun priority:

1. `principle_expanded` for Qwen/Llama/Gemma with 100 samples per task
2. `length_control_expanded` for Qwen/Llama/Gemma
3. `taxonomy_controls_expanded` for Qwen/Llama/Gemma

The first paper-backed seen/unseen selection experiment does not need rerun because it already used paper-backed prompts.

---

## 9. Claims I want you to audit

Please judge each claim as:

- Strongly supported
- Moderately supported
- Weakly supported
- Not supported / overclaim

Claims:

1. Seen-task prompt performance transfers imperfectly but measurably to unseen tasks.
2. Activation signatures contain diagnostic signal about prompt transferability.
3. Activation signatures are not currently demonstrated as causal steering vectors.
4. Minimal-answer / verbosity-control prompts are more stable than heavy reasoning/expert prompts in this setup.
5. Explicit final-answer boundary helps some models but is not universal.
6. The observed effect is not simply “prompt is shorter” but relates to generated answer mode.
7. Cross-model alignment suggests shared behavior-level prompting principles but model-specific internal implementation.
8. This is a credible undergraduate research / workshop-level project if claims are phrased conservatively.

---

## 10. Known weaknesses

Please pay special attention to these:

- sample sizes are limited
- task subsets are mini/sampled
- output length and answer parsing can confound accuracy
- shorter output may improve exact-match extraction without improving reasoning
- activation R2 may reflect prompt lexical/style features rather than deep behavior
- patching failure weakens causal interpretation
- direct cross-model vector comparison is limited by different hidden dimensions
- old controlled results and new ref-aligned rerun must not be mixed as if they were the same experiment

---

## 11. Desired output from you

Please produce a critical review in this exact structure:

```text
1. Verdict in one paragraph
2. Claim-by-claim support table
3. Biggest methodological risks
4. What should be rewritten in the final report
5. What experiments are unnecessary
6. What experiments are essential before submission
7. Best conservative final thesis
8. Whether this is publishable as undergraduate/workshop research
```

Do not be polite for the sake of politeness. I want the most useful skeptical version.
