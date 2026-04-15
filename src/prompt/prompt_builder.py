from __future__ import annotations


def build_messages(system_prompt: str, user_input: str):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]


def render_chat_prompt(tokenizer, system_prompt: str, user_input: str, add_generation_prompt: bool = True) -> str:
    messages = build_messages(system_prompt, user_input)
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
    return (
        f"<|system|>\n{system_prompt}\n"
        f"<|user|>\n{user_input}\n"
        f"<|assistant|>\n"
    )


def build_input(tokenizer, system_prompt: str, user_input: str) -> str:
    return render_chat_prompt(
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_input=user_input,
        add_generation_prompt=True,
    )


def _locate_span(text: str, fragment: str, search_from: int = 0) -> tuple[int, int]:
    start = text.find(fragment, search_from)
    if start < 0:
        raise ValueError(f"Could not find fragment in rendered prompt: {fragment!r}")
    return start, start + len(fragment)


def _char_to_token_index(offsets, char_index: int) -> int:
    for token_index, (start, end) in enumerate(offsets):
        if start <= char_index < end:
            return token_index
    raise ValueError(f"Could not map character index {char_index} to a token.")


def locate_token_positions(tokenizer, rendered_prompt: str, system_prompt: str, user_input: str):
    system_start, system_end = _locate_span(rendered_prompt, system_prompt)
    user_start, user_end = _locate_span(rendered_prompt, user_input, search_from=system_end)

    encoding = tokenizer(
        rendered_prompt,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    offsets = encoding["offset_mapping"]

    return {
        "system_last_token": _char_to_token_index(offsets, system_end - 1),
        "first_user_token": _char_to_token_index(offsets, user_start),
        "system_char_span": (system_start, system_end),
        "user_char_span": (user_start, user_end),
    }

