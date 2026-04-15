from __future__ import annotations

import re


ROLE_PREFIXES = [
    "Act as",
    "Serve as",
    "Take the role of",
]

SYNONYM_MAPS = [
    {
        "answer": "respond",
        "accurately": "correctly",
        "careful": "deliberate",
        "carefully": "deliberately",
        "verify": "check",
        "final answer": "answer",
        "provide": "give",
        "double-check": "check again",
        "avoid": "steer clear of",
        "clear": "easy-to-follow",
    },
    {
        "answer": "reply",
        "careful": "thoughtful",
        "carefully": "thoughtfully",
        "track": "monitor",
        "verify": "cross-check",
        "ensure": "make sure",
        "concise": "brief",
        "critical": "skeptical",
        "final answer": "final response",
    },
    {
        "answer": "respond",
        "careful": "rigorous",
        "carefully": "rigorously",
        "organize": "structure",
        "check": "validate",
        "verify": "validate",
        "avoid": "prevent",
        "clear": "well-structured",
        "helpful": "useful",
    },
]


def split_sentences(text: str):
    pieces = re.split(r"(?<=[.!?])\s+", text.strip())
    return [piece.strip() for piece in pieces if piece.strip()]


def rewrite_role_sentence(sentence: str, variant_index: int) -> str:
    prefixes = ROLE_PREFIXES
    prefix = prefixes[variant_index % len(prefixes)]
    match = re.match(r"^You are\s+(.*)$", sentence.strip(), flags=re.IGNORECASE)
    if match:
        body = match.group(1).strip()
        return f"{prefix} {body}"
    return sentence


def apply_synonym_map(text: str, synonym_map: dict[str, str]) -> str:
    updated = text
    for source, target in sorted(synonym_map.items(), key=lambda item: len(item[0]), reverse=True):
        updated = re.sub(
            rf"\b{re.escape(source)}\b",
            target,
            updated,
            flags=re.IGNORECASE,
        )
    return updated


def normalize_sentence(sentence: str) -> str:
    sentence = re.sub(r"\s+", " ", sentence).strip()
    if sentence and sentence[-1] not in ".!?":
        sentence += "."
    return sentence


def heuristic_paraphrase(text: str, variant_index: int) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return text

    synonym_map = SYNONYM_MAPS[variant_index % len(SYNONYM_MAPS)]
    rewritten = []
    for index, sentence in enumerate(sentences):
        candidate = sentence
        if index == 0:
            candidate = rewrite_role_sentence(candidate, variant_index)
        candidate = apply_synonym_map(candidate, synonym_map)
        rewritten.append(normalize_sentence(candidate))

    if len(rewritten) > 2 and variant_index == 1:
        rewritten = [rewritten[0], *rewritten[2:], rewritten[1]]
    elif len(rewritten) > 2 and variant_index == 2:
        rewritten = [rewritten[0], rewritten[-1], *rewritten[1:-1]]

    return " ".join(rewritten)


def build_paraphrase_records(prompt_record: dict, num_paraphrases: int = 3):
    if prompt_record.get("variant", "original") != "original":
        raise ValueError("Paraphrases should be generated from original prompt variants only.")

    group_id = prompt_record.get("group_id", prompt_record["id"])
    results = [dict(prompt_record)]
    for variant_index in range(num_paraphrases):
        variant_name = f"paraphrase_{variant_index + 1}"
        paraphrase_text = heuristic_paraphrase(prompt_record["text"], variant_index)
        if paraphrase_text == prompt_record["text"]:
            paraphrase_text = f"{ROLE_PREFIXES[variant_index % len(ROLE_PREFIXES)]} the following policy. {prompt_record['text']}"
        results.append(
            {
                **prompt_record,
                "id": f"{group_id}__{variant_name}",
                "group_id": group_id,
                "variant": variant_name,
                "source": f"{prompt_record['source']}_paraphrase",
                "text": paraphrase_text,
                "parent_id": prompt_record["id"],
            }
        )
    return results

