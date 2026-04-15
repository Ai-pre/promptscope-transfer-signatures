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
python scripts/build_paraphrase_prompt_pool.py --limit-groups 10
```

This writes `data/prompts_paraphrase_pilot.jsonl` with 10 original prompt groups and 3 paraphrases per group.

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

- 20 original prompts
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
