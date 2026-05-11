# GPT Deep-Research Review 반영 액션 플랜

이 문서는 `deep-research-report (2).md`의 결론을 현재 repo 작업으로 변환한 실행 체크리스트다.

## 1. 가장 중요한 결론

현재 연구는 바로 “causal mechanism을 발견했다”라고 쓰면 과장이다. 가장 안전한 포지셔닝은 다음이다.

> system prompt가 만드는 activation signature는 unseen transfer를 담는 diagnostic signal이며, 현재까지 가장 반복적으로 보이는 controlled principle은 minimal answer mode / verbosity control이다.

논문/보고서에서 피해야 할 표현:

- `improves general capability`
- `discovers the causal prompt vector`
- `concise direction mediates transfer`
- `universal cross-model direction`

권장 표현:

- `predicts transferable prompting behavior`
- `diagnostic activation signature`
- `boundary-localized candidate signal`
- `minimal-answer / verbosity-control mode`
- `causal claim remains limited unless patching succeeds`

## 2. 지금 당장 반영한 것

- `README.md`에 Prompt Report taxonomy와 source-derived prompt controls를 추가했다.
- `data/prompts_principle_boundary.jsonl`을 ref-aligned wording으로 교체했다.
- `data/prompts_length_control_boundary.jsonl`을 ref-aligned wording으로 교체했다.
- `data/prompts_taxonomy_controls.jsonl`을 추가했다.
- `scripts/build_principle_boundary_prompt_pool.py`와 `scripts/build_length_control_prompt_pool.py`도 ref-aligned wording을 재생성하도록 수정했다.
- `scripts/run_length_confound_analysis.py`를 추가했다.

## 3. 재실험 우선순위

### Priority 1. Ref-aligned principle decomposition

목적:

- `concise`, `format`, `check`, `soft_reason`, `careful` 효과가 reference-backed wording에서도 유지되는지 확인한다.

권장 샘플:

- `--limit-per-task 100`

명령:

```bash
python scripts/run_eval.py --config configs/config.unsloth_qwen3_4b_principle_expanded.yaml --limit-per-task 100
python scripts/extract_activation.py --config configs/config.unsloth_qwen3_4b_principle_expanded.yaml --limit-per-task 100
python scripts/run_principle_analysis.py --config configs/config.unsloth_qwen3_4b_principle_expanded.yaml --reference-config configs/config.unsloth_qwen3_4b_general_expanded.yaml

python scripts/run_eval.py --config configs/config.unsloth_llama31_8b_principle_expanded.yaml --limit-per-task 100
python scripts/extract_activation.py --config configs/config.unsloth_llama31_8b_principle_expanded.yaml --limit-per-task 100
python scripts/run_principle_analysis.py --config configs/config.unsloth_llama31_8b_principle_expanded.yaml --reference-config configs/config.unsloth_llama31_8b_general_expanded.yaml

python scripts/run_eval.py --config configs/config.unsloth_gemma4_e2b_principle_expanded.yaml --limit-per-task 100
python scripts/extract_activation.py --config configs/config.unsloth_gemma4_e2b_principle_expanded.yaml --limit-per-task 100
python scripts/run_principle_analysis.py --config configs/config.unsloth_gemma4_e2b_principle_expanded.yaml --reference-config configs/config.unsloth_gemma4_e2b_general_expanded.yaml
```

### Priority 2. Length / decoding confound control

목적:

- `concise`/`short` 효과가 단순 output length, prompt length, final-answer marker 때문에 생긴 것인지 통제한다.

명령:

```bash
python scripts/run_eval.py --config configs/config.unsloth_qwen3_4b_length_control_expanded.yaml --limit-per-task 100
python scripts/run_length_control_analysis.py --config configs/config.unsloth_qwen3_4b_length_control_expanded.yaml
python scripts/run_length_confound_analysis.py --config configs/config.unsloth_qwen3_4b_length_control_expanded.yaml

python scripts/run_eval.py --config configs/config.unsloth_llama31_8b_length_control_expanded.yaml --limit-per-task 100
python scripts/run_length_control_analysis.py --config configs/config.unsloth_llama31_8b_length_control_expanded.yaml
python scripts/run_length_confound_analysis.py --config configs/config.unsloth_llama31_8b_length_control_expanded.yaml

python scripts/run_eval.py --config configs/config.unsloth_gemma4_e2b_length_control_expanded.yaml --limit-per-task 100
python scripts/run_length_control_analysis.py --config configs/config.unsloth_gemma4_e2b_length_control_expanded.yaml
python scripts/run_length_confound_analysis.py --config configs/config.unsloth_gemma4_e2b_length_control_expanded.yaml
```

새 confound 분석 결과 파일:

- `length_confound_results/length_confound_summary.json`
- `length_confound_results/length_confound_component_effects.json`
- `length_confound_results/length_confound_task_component_effects.json`
- `length_confound_results/length_confound_report.md`

해석 주의:

- output length와 final-answer marker는 post-treatment variable이므로 이것은 causal adjustment가 아니라 sensitivity analysis다.
- adjusted effect가 남으면 “단순 길이/marker만으로는 환원되지 않는다” 정도까지 주장할 수 있다.
- adjusted effect가 사라지면 “minimal answer effect는 대부분 length/format-mediated”라고 솔직히 써야 한다.

### Priority 3. Prompt Report taxonomy controls

목적:

- `answer_only`, `structured_output`, `zero_shot_cot`, `generated_knowledge`, `least_to_most`, `expert`, `emotion`, `meta_prompting`, `promptwizard_like`를 같은 controlled setup에서 비교한다.

명령:

```bash
python scripts/run_eval.py --config configs/config.unsloth_qwen3_4b_taxonomy_controls_expanded.yaml --limit-per-task 100
python scripts/run_length_control_analysis.py --config configs/config.unsloth_qwen3_4b_taxonomy_controls_expanded.yaml
python scripts/run_length_confound_analysis.py --config configs/config.unsloth_qwen3_4b_taxonomy_controls_expanded.yaml

python scripts/run_eval.py --config configs/config.unsloth_llama31_8b_taxonomy_controls_expanded.yaml --limit-per-task 100
python scripts/run_length_control_analysis.py --config configs/config.unsloth_llama31_8b_taxonomy_controls_expanded.yaml
python scripts/run_length_confound_analysis.py --config configs/config.unsloth_llama31_8b_taxonomy_controls_expanded.yaml

python scripts/run_eval.py --config configs/config.unsloth_gemma4_e2b_taxonomy_controls_expanded.yaml --limit-per-task 100
python scripts/run_length_control_analysis.py --config configs/config.unsloth_gemma4_e2b_taxonomy_controls_expanded.yaml
python scripts/run_length_confound_analysis.py --config configs/config.unsloth_gemma4_e2b_taxonomy_controls_expanded.yaml
```

## 4. Patching 결과 해석 방침

기존 patching은 hook이 실행되고 direction이 적용되었지만 behavior 변화가 거의 없었다. 이 결과는 실패가 아니라 claim boundary를 정하는 데 중요하다.

쓸 수 있는 말:

- activation signature is diagnostic
- single-layer/single-token component-direction injection did not reproduce the prompt behavior
- localization and editing/steering should be separated

아직 쓰면 안 되는 말:

- concise direction causally mediates transfer
- boundary state has been causally identified
- injecting the direction improves unseen accuracy

## 5. 다음 patching을 다시 한다면

GPT review가 요구한 stronger protocol:

- clean run: `concise` 또는 observed best prompt
- corrupted run: matched prompt without component
- patch locations: `system_last_token`, `first_user_token`
- layers: predictive peak 주변 2개 + wrong-layer controls 2개
- controls: wrong-position, wrong-layer, wrong-example, random matched-norm, length-only, lexical paraphrase
- metrics: accuracy, output length, format compliance, answer logit margin

현재 repo의 patching script는 accuracy/length 중심이다. answer logit margin까지 넣으려면 generation 전 첫-answer-token logit extraction이 추가로 필요하다.

## 6. 최종 claim 계층

### 지금도 방어 가능한 claim

- Activation signatures contain diagnostic signal about prompt-level unseen transfer.
- Signal often concentrates near system/user boundary positions.
- Minimal answer / verbosity-control prompts are recurring strong candidates in the current setup.
- Patching results do not yet support a causal steering-vector interpretation.

### ref-aligned rerun 후 방어 가능한 claim

- The minimal-answer/verbosity-control pattern survives reference-backed prompt wording.
- The effect is not fully reducible to prompt taxonomy arbitrariness.
- If confound-adjusted effects remain positive, it is not fully reducible to observed output length or marker use.

### 아직 추가 실험 없이는 방어 불가능한 claim

- The concise direction causally mediates unseen transfer.
- Boundary activation state is sufficient to reproduce prompt behavior.
- A universal cross-family direction exists.

## 7. 보고서/논문 구조

권장 구조:

1. Introduction: prompt engineering은 behavior 중심이고, interpretability는 prompt principle 중심이 약하다는 gap.
2. Related Work: activation steering/patching, function vectors, prompt sensitivity, CoT faithfulness, instruction tuning limitations, Prompt Report taxonomy.
3. Method: activation signature, seen/unseen transfer, boundary positions, prompt pools, component directions.
4. Observational Results: paper-backed prompt transfer prediction, robustness, cross-model alignment.
5. Controlled Prompt Decomposition: ref-aligned principle and taxonomy results.
6. Length/Decoding Confound Analysis: length-control and adjusted effects.
7. Causal Tests: patching negative/limited result, claim boundary.
8. Discussion: diagnostic signal vs causal mechanism, minimal answer mode, future patching protocol.

## 8. 최종 thesis 후보

가장 안전한 thesis:

> System prompts that transfer across tasks leave measurable activation signatures near prompt/user boundary states. These signatures are useful diagnostic signals for transferable prompting behavior, but current causal patching does not justify treating them as standalone steering vectors. Ref-aligned controlled decompositions suggest that a minimal-answer / verbosity-control mode is a stronger recurring explanation than heavy reasoning or expert scaffolding.

좀 더 짧은 thesis:

> Activation signatures diagnose transferable system-prompt behavior; in our controlled setting, the most stable principle is minimal answer mode rather than heavy reasoning scaffolding.
