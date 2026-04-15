# PromptScope: Transferable System Prompt Signatures

This repository implements an end-to-end pilot pipeline for testing whether transferable system prompts produce shared activation signatures and whether those signatures predict unseen-task transfer.

## What is included

- `scripts/run_eval.py`: runs prompt-level generation and task evaluation
- `scripts/extract_activation.py`: extracts hidden states and computes `delta_h = h(prompt) - h(base_prompt)`
- `scripts/run_analysis.py`: computes similarity, stability, and transfer prediction baselines
- `data/prompts.jsonl`: starter prompt pool with 30 canonical prompts
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
  "id": "manual_helpful",
  "group_id": "manual_helpful",
  "variant": "original",
  "source": "manual",
  "text": "You are a helpful assistant."
}
```

To support paraphrase stability experiments, add extra rows with the same `group_id` and different `variant` values such as `paraphrase_1`, `paraphrase_2`, `paraphrase_3`.

## Run

```bash
python scripts/run_eval.py
python scripts/extract_activation.py
python scripts/run_analysis.py
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
