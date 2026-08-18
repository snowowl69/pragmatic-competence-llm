"""
Instruction-format data converter for QLoRA fine-tuning.

Converts benchmark items into instruction-response pairs that match
the base model's chat template. Implements **Approach B** from the
execution guide (Section 14): targets include both the gold label AND
a brief reasoning/intended-meaning explanation, not just a bare label.

This is what makes the fine-tuning "pragmatics-aware" — the model learns
to reason about speaker intent, not just classify.

Usage:
    python src/training/data_formatter.py \\
        --input data/benchmark/train.jsonl \\
        --output data/training/train_instructions.jsonl \\
        [--categories sarcasm,implicature,indirect_speech_act]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Instruction templates per category (Approach B — label + reasoning)
# ---------------------------------------------------------------------------

SARCASM_INSTRUCTION = """Analyze the following utterance in context. Determine whether the speaker is being sincere or sarcastic, and explain the intended meaning.

Context: {context}
Speaker: {speaker}
Utterance: "{utterance}"

Respond with:
1. Label: sincere or sarcastic
2. Intended meaning: what the speaker actually means"""

SARCASM_RESPONSE = """Label: {label}
Intended meaning: {intended_meaning}"""

IMPLICATURE_INSTRUCTION = """In the following exchange, B gives an indirect reply to A's question. Determine what B is implying and explain the intended meaning.

{context}
B: "{utterance}"

Respond with:
1. Implied answer: yes, no, or uncertain
2. Intended meaning: what B is actually communicating"""

IMPLICATURE_RESPONSE = """Implied answer: {label}
Intended meaning: {intended_meaning}"""

INDIRECT_INSTRUCTION = """Analyze the following utterance in context. The utterance may function differently from its grammatical form. Determine the speaker's actual intent.

Context: {context}
Speaker: {speaker}
Utterance: "{utterance}"

Respond with:
1. Actual intent: statement, request, question, suggestion, or complaint
2. Intended meaning: what the speaker is really trying to do"""

INDIRECT_RESPONSE = """Actual intent: {label}
Intended meaning: {intended_meaning}"""


# ---------------------------------------------------------------------------
# Formatting logic
# ---------------------------------------------------------------------------

INSTRUCTION_TEMPLATES = {
    "sarcasm": (SARCASM_INSTRUCTION, SARCASM_RESPONSE),
    "implicature": (IMPLICATURE_INSTRUCTION, IMPLICATURE_RESPONSE),
    "indirect_speech_act": (INDIRECT_INSTRUCTION, INDIRECT_RESPONSE),
}


def format_instruction_pair(item: dict[str, Any]) -> dict[str, str]:
    """Convert a benchmark item to an instruction-response pair.

    Returns:
        {"instruction": str, "input": str, "output": str}
        where input is empty (all info is in the instruction).
    """
    category = item["pragmatic_category"]
    if category not in INSTRUCTION_TEMPLATES:
        raise ValueError(f"Unknown category: {category}")

    instr_template, resp_template = INSTRUCTION_TEMPLATES[category]

    instruction = instr_template.format(
        context=item["context"],
        speaker=item.get("speaker", "unknown"),
        utterance=item["utterance"],
    )

    response = resp_template.format(
        label=item["label"],
        intended_meaning=item["intended_meaning"],
    )

    return {
        "instruction": instruction,
        "input": "",
        "output": response,
    }


def format_chat_template(
    item: dict[str, Any],
    system_message: str = "You are a pragmatics expert. Analyze utterances to determine their intended meaning beyond the literal words.",
) -> list[dict[str, str]]:
    """Convert a benchmark item to chat-format messages.

    Returns a list of message dicts compatible with HuggingFace chat templates.
    """
    pair = format_instruction_pair(item)

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": pair["instruction"]},
        {"role": "assistant", "content": pair["output"]},
    ]

    return messages


def format_for_sft(
    item: dict[str, Any],
    tokenizer: Any = None,
) -> str:
    """Format an item as a single text string for SFTTrainer.

    If a tokenizer is provided, uses its chat template.
    Otherwise, uses a generic instruction format.
    """
    messages = format_chat_template(item)

    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(messages, tokenize=False)

    # Generic fallback format
    pair = format_instruction_pair(item)
    return f"### Instruction:\n{pair['instruction']}\n\n### Response:\n{pair['output']}"


# ---------------------------------------------------------------------------
# Batch processing
# ---------------------------------------------------------------------------


def convert_benchmark_to_training_data(
    input_file: str | Path,
    output_file: str | Path,
    categories: list[str] | None = None,
    format_type: str = "instruction",  # "instruction" or "chat"
) -> int:
    """Convert a benchmark JSONL file to training-format JSONL.

    Args:
        input_file: Path to benchmark JSONL (e.g., data/benchmark/train.jsonl)
        output_file: Path to output training JSONL
        categories: Optional filter — only include these categories
        format_type: "instruction" for instruction/input/output format,
                     "chat" for messages format

    Returns:
        Number of items converted
    """
    input_file = Path(input_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    skipped = 0

    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:

        for line in fin:
            item = json.loads(line.strip())

            # Filter by category if specified
            if categories and item["pragmatic_category"] not in categories:
                skipped += 1
                continue

            # Skip ambiguous items — they don't have a clear target
            if item.get("label") == "ambiguous":
                skipped += 1
                continue

            if format_type == "chat":
                messages = format_chat_template(item)
                training_item = {"messages": messages, "id": item["id"]}
            else:
                pair = format_instruction_pair(item)
                training_item = {**pair, "id": item["id"]}

            fout.write(json.dumps(training_item, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Converted {count} items ({skipped} skipped) → {output_file}")
    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert benchmark data to instruction-format training data."
    )
    parser.add_argument("--input", type=str,
                        default=str(PROJECT_ROOT / "data" / "benchmark" / "train.jsonl"),
                        help="Input benchmark JSONL file")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "training" / "train_instructions.jsonl"),
                        help="Output training JSONL file")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated list of categories to include (default: all)")
    parser.add_argument("--format", type=str, default="instruction",
                        choices=["instruction", "chat"],
                        help="Output format type")

    args = parser.parse_args()

    categories = args.categories.split(",") if args.categories else None
    cat_label = ",".join(categories) if categories else "all"
    print(f"Converting benchmark to training format (categories: {cat_label})...")

    convert_benchmark_to_training_data(
        input_file=args.input,
        output_file=args.output,
        categories=categories,
        format_type=args.format,
    )


if __name__ == "__main__":
    main()
