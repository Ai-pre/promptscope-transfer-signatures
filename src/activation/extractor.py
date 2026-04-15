from __future__ import annotations

import torch

from src.model.load_model import get_model_device
from src.prompt.prompt_builder import locate_token_positions


def extract_hidden_states(model, tokenizer, rendered_prompt: str):
    device = get_model_device(model)
    inputs = tokenizer(
        rendered_prompt,
        return_tensors="pt",
        add_special_tokens=False,
    )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    return outputs.hidden_states, inputs


def get_positions(tokenizer, rendered_prompt: str, system_prompt: str, user_input: str):
    positions = locate_token_positions(
        tokenizer=tokenizer,
        rendered_prompt=rendered_prompt,
        system_prompt=system_prompt,
        user_input=user_input,
    )
    return {
        "system_last_token": positions["system_last_token"],
        "first_user_token": positions["first_user_token"],
    }


def resolve_layer_index(hidden_states, layer_spec: int) -> int:
    num_hidden_states = len(hidden_states)
    resolved = num_hidden_states + layer_spec if layer_spec < 0 else layer_spec
    if resolved < 0 or resolved >= num_hidden_states:
        raise IndexError(
            f"Layer index {layer_spec} resolved to {resolved}, "
            f"but hidden states only has {num_hidden_states} entries."
        )
    return resolved


def select_hidden_vector(hidden_states, layer_spec: int, token_index: int):
    resolved_layer = resolve_layer_index(hidden_states, layer_spec)
    return resolved_layer, hidden_states[resolved_layer][0, token_index]


def compute_delta_h(h_prompt, h_base):
    return h_prompt - h_base

