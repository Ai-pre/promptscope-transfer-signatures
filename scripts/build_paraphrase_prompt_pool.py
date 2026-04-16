from __future__ import annotations

import argparse
import json

from _bootstrap import bootstrap_project_root

bootstrap_project_root()

from src.prompt.paraphrase import build_paraphrase_records
from src.utils.io import ensure_dir, load_prompts, resolve_path


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a prompt pool with 3 paraphrases per original prompt."
    )
    parser.add_argument("--input", default="data/prompts_paper_backed.jsonl")
    parser.add_argument("--output", default="data/prompts_paper_backed_paraphrase.jsonl")
    parser.add_argument("--num-paraphrases", type=int, default=3)
    parser.add_argument(
        "--limit-groups",
        type=int,
        default=10,
        help="Optional cap on the number of original prompt groups to include.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    project_root = bootstrap_project_root()
    input_path = resolve_path(project_root, args.input)
    output_path = resolve_path(project_root, args.output)
    ensure_dir(output_path.parent)

    original_prompts = [
        prompt for prompt in load_prompts(input_path) if prompt.get("variant", "original") == "original"
    ]
    if args.limit_groups is not None:
        original_prompts = original_prompts[: args.limit_groups]

    records = []
    for prompt_record in original_prompts:
        records.extend(build_paraphrase_records(prompt_record, num_paraphrases=args.num_paraphrases))

    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(
        f"[build_paraphrase_prompt_pool] wrote {len(records)} prompt variants "
        f"across {len(original_prompts)} groups to {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
