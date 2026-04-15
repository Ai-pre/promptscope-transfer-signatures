from __future__ import annotations

import importlib.util
import warnings

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


def has_accelerate() -> bool:
    return importlib.util.find_spec("accelerate") is not None


def resolve_runtime_device_map(device_map):
    if device_map in {None, "", "none"}:
        return None

    if not has_accelerate():
        warnings.warn(
            "accelerate is not installed, so device_map will be disabled and the model "
            "will be loaded onto a single device.",
            RuntimeWarning,
        )
        return None

    if not torch.cuda.is_available() and device_map == "auto":
        warnings.warn(
            "CUDA is not available, so device_map='auto' will be disabled and the model "
            "will be loaded on CPU.",
            RuntimeWarning,
        )
        return None

    return device_map


def maybe_adjust_dtype_for_device(torch_dtype, runtime_device_map):
    if torch_dtype == "auto":
        return torch_dtype

    loading_on_cpu = runtime_device_map is None and not torch.cuda.is_available()
    if loading_on_cpu and torch_dtype in {torch.float16, torch.bfloat16}:
        warnings.warn(
            "Half-precision dtype requested without CUDA support; switching to float32 "
            "for safer CPU execution.",
            RuntimeWarning,
        )
        return torch.float32

    return torch_dtype


def load_model(model_name: str, torch_dtype: str = "auto", device_map: str | None = "auto"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
    tokenizer.padding_side = "left"

    requested_dtype = parse_torch_dtype(torch_dtype)
    runtime_device_map = resolve_runtime_device_map(device_map)
    runtime_dtype = maybe_adjust_dtype_for_device(requested_dtype, runtime_device_map)

    model_kwargs = {
        "output_hidden_states": True,
        "torch_dtype": runtime_dtype,
        "trust_remote_code": True,
    }
    if runtime_device_map is not None:
        model_kwargs["device_map"] = runtime_device_map

    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    if runtime_device_map is None:
        target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(target_device)

    model.eval()
    return model, tokenizer
