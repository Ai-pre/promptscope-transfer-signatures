from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


TORCH_DTYPES = {
    "auto": "auto",
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


def parse_torch_dtype(dtype_name: str | None):
    if dtype_name is None:
        return "auto"
    if dtype_name not in TORCH_DTYPES:
        raise ValueError(
            f"Unsupported torch_dtype={dtype_name!r}. "
            f"Expected one of {sorted(TORCH_DTYPES)}."
        )
    return TORCH_DTYPES[dtype_name]


def get_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def load_model(model_name: str, torch_dtype: str = "auto", device_map: str = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
        torch_dtype=parse_torch_dtype(torch_dtype),
        device_map=device_map,
        trust_remote_code=True,
    )
    model.eval()
    return model, tokenizer

