# 랩미팅 최종 정리

## 1. Motivation

처음 질문은 단순히 어떤 system prompt가 성능이 좋은지가 아니었음. 더 궁금한 건 왜 어떤 prompt는 seen task에서만 잘하는 게 아니라 unseen task에서도 성능이 유지되는가였음.

그래서 transferable prompt를 seen task로 고른 뒤에도 unseen task에서 성능이 유지되는 prompt로 정의하고, 그 prompt들이 모델 내부에서 공통 activation signature를 만드는지 보려고 했음.

최종 목표는 activation을 이용해 transferable prompting의 principle을 찾는 것.

## 2. Method

모델은 세 계열을 사용함.

| Model family | Model |
|---|---|
| Qwen | `Qwen/Qwen2.5-7B-Instruct` |
| Llama family | `NousResearch/Hermes-3-Llama-3.1-8B` |
| Gemma family | `FuseAI/FuseChat-Gemma-2-9B-SFT` |

Task는 seen/unseen으로 나눔.

| Split | Tasks |
|---|---|
| Seen | `GSM8K`, `CSQA` |
| Unseen | `SVAMP`, `BoolQ` |

각 prompt에 대해 성능과 activation을 같이 측정함.

Activation은 `delta_h = h(prompt) - h(base_prompt)`로 정의함.

Base prompt는 `You are a helpful assistant.`

Activation 위치는 `system_last_token`, `first_user_token`을 봄.

Layer는 early/middle/late를 봄.

실험은 두 단계로 진행함.

첫째, 논문/공식 repo 기반 prompt pool로 reference 실험을 돌려 activation이 transfer를 예측하는지 확인함.

둘째, reference 결과를 바탕으로 principle probe를 만들어 `concise`, `careful`, `format`, `check`, `soft_reason` 같은 요소를 분해해서 봄.

## 3. Results

### 3-1. Reference 실험

세 모델 계열 모두에서 activation은 transfer-related signal을 가졌음.

| Model | Activation R2 | Activation top-k unseen | Seen top-k unseen | Random top-k unseen |
|---|---:|---:|---:|---:|
| Qwen2.5-7B | 0.438 | 0.585 | 0.591 | 0.480 |
| Hermes-Llama | 0.405 | 0.789 | 0.802 | 0.644 |
| FuseChat-Gemma | 0.256 | 0.640 | 0.600 | 0.543 |

해석하면 activation은 random보다 훨씬 좋은 signal이었고, 일부 모델에서는 seen-task baseline보다도 좋은 selection 결과가 나왔음. 다만 모든 모델에서 seen baseline을 일관되게 이긴 것은 아님.

또 중요한 점은 strongest activation slice가 반복적으로 user-input boundary 쪽에서 나왔다는 것임.

Qwen에서는 `layer 28 + first_user_token`, Hermes에서는 `first_user_token` 계열 slice, FuseChat-Gemma에서도 user-boundary slice가 강하게 나왔음.

즉 transferable prompt는 모델 전체를 크게 바꾸는 것보다, user question을 읽는 지점의 internal state를 바꾸는 것에 더 가까워 보임.

### 3-2. Principle probe 실험

Principle probe에서는 어떤 prompt 요소가 실제 transfer에 도움 되는지 봄.

| Model | Best unseen probe | Best similarity probe |
|---|---|---|
| Qwen2.5-7B | `principle3_concise_careful_format` | `principle3_concise_careful_format` |
| Hermes-Llama | `principle3_careful_format_check` | `principle3_soft_reason` |
| FuseChat-Gemma | `principle3_concise_careful_format` | `principle3_soft_reason_format` |

Qwen과 FuseChat-Gemma에서는 `concise + careful + format` 조합이 성능 최고로 반복됨.

Hermes에서는 정확한 best 조합은 달랐지만, 여전히 heavy scaffold가 아니라 lightweight boundary-setting family 안에서 best가 나왔음.

따라서 하나의 universal prompt 문장이 있다기보다는, transferable prompt family가 있는 것으로 보는 게 더 맞음.

## 4. Conclusion

현재 가장 강한 결론은 다음과 같음.

Transferable prompt의 원리는 heavy expert/reasoning scaffold가 아니라 lightweight boundary-setting에 가까움.

특히 `concise`, `careful`, `minimal format contract` 계열이 가장 반복적으로 유망했음.

반대로 강한 expert persona, hard step-by-step reasoning, verbose scaffold, multi-agent 구조는 이전 probe에서 일관되게 약하거나 해로웠음.

Activation은 mechanism을 설명하는 데는 유용했지만, raw centroid cosine을 그대로 prompt selection score로 쓰는 것은 실패했음.

최종 한 줄 결론:

잘 전이되는 프롬프트는 모델에게 많은 절차를 강제하는 프롬프트가 아니라, 질문을 읽고 답하는 경계를 짧고 명확하게 잡아주는 프롬프트다.

