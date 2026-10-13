"""
Dataset normalization pipeline.

Converts each raw source dataset into the unified benchmark schema
(see schema.py). Each source has its own parser function that maps
source-specific fields to the canonical schema fields.

Usage:
    python src/preprocessing/normalize.py [--all | --mustard | --isarcasm | --circa | --indirect]

Output: data/processed/*.jsonl (one file per source, in unified schema)
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.schema import validate_item, print_validation_report
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# ---------------------------------------------------------------------------
# MUStARD normalization
# ---------------------------------------------------------------------------


def normalize_mustard() -> list[dict[str, Any]]:
    """Normalize MUStARD sarcasm dataset to unified schema.

    MUStARD's sarcasm_data.json has entries keyed by utterance ID, with fields:
    - "sarcasm": bool
    - "utterance": str
    - "context": list of prior utterances (dialogue context)
    - "speaker": str
    """
    source_path = RAW_DATA_DIR / "mustard" / "sarcasm_data.json"
    hf_path = RAW_DATA_DIR / "mustard" / "mustard_hf.jsonl"

    items: list[dict[str, Any]] = []

    if source_path.exists():
        print(f"[MUStARD] Loading from {source_path}")
        with open(source_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for idx, (key, entry) in enumerate(raw_data.items()):
            # Build context from dialogue history
            context_parts = entry.get("context", [])
            if isinstance(context_parts, list):
                context = " ".join(str(c) for c in context_parts) if context_parts else "No prior context."
            else:
                context = str(context_parts)

            utterance = entry.get("utterance", "")
            is_sarcastic = entry.get("sarcasm", False)
            speaker = entry.get("speaker", "unknown")

            item = {
                "id": f"sarc_{idx + 1:04d}",
                "context": context.strip() or "No prior context.",
                "speaker": speaker or "unknown",
                "utterance": utterance,
                "literal_meaning": f"The literal content of: \"{utterance}\"",
                "intended_meaning": (
                    f"The speaker is being sarcastic — the intended meaning opposes the literal reading."
                    if is_sarcastic
                    else f"The speaker sincerely means: \"{utterance}\""
                ),
                "pragmatic_category": "sarcasm",
                "label": "sarcastic" if is_sarcastic else "sincere",
                "answer_options": ["sincere", "sarcastic"],
                "difficulty": "medium",  # Default; to be refined during annotation
                "requires_context": True,
                "source": "MUStARD",
                "license": "research-use",
                "split": "train",  # Will be reassigned during splitting
                "annotator_notes": f"Original key: {key}",
            }
            items.append(item)

    elif hf_path.exists():
        print(f"[MUStARD] Loading from HF download {hf_path}")
        with open(hf_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                entry = json.loads(line.strip())
                utterance = entry.get("utterance", entry.get("text", ""))
                is_sarcastic = entry.get("sarcasm", entry.get("label", 0))
                if isinstance(is_sarcastic, int):
                    is_sarcastic = bool(is_sarcastic)

                context = entry.get("context", "No prior context.")
                if isinstance(context, list):
                    context = " ".join(str(c) for c in context)

                item = {
                    "id": f"sarc_{idx + 1:04d}",
                    "context": str(context).strip() or "No prior context.",
                    "speaker": entry.get("speaker", "unknown"),
                    "utterance": str(utterance),
                    "literal_meaning": f"The literal content of: \"{utterance}\"",
                    "intended_meaning": (
                        "The speaker is being sarcastic."
                        if is_sarcastic
                        else f"The speaker sincerely means: \"{utterance}\""
                    ),
                    "pragmatic_category": "sarcasm",
                    "label": "sarcastic" if is_sarcastic else "sincere",
                    "answer_options": ["sincere", "sarcastic"],
                    "difficulty": "medium",
                    "requires_context": True,
                    "source": "MUStARD",
                    "license": "research-use",
                    "split": "train",
                    "annotator_notes": "",
                }
                items.append(item)
    else:
        print(f"[MUStARD] ✗ No source file found. Run download.py first.")
        print(f"  Expected: {source_path} or {hf_path}")
        return []

    print(f"[MUStARD] ✓ Normalized {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# iSarcasm normalization
# ---------------------------------------------------------------------------


def normalize_isarcasm() -> list[dict[str, Any]]:
    """Normalize iSarcasm dataset to unified schema.

    iSarcasm CSV has columns: tweet text, sarcasm label (0/1), and optionally
    irony type. Context is limited (tweet-only), so requires_context = False
    for most items.
    """
    items: list[dict[str, Any]] = []
    hf_path = RAW_DATA_DIR / "isarcasm" / "isarcasm_hf.jsonl"
    csv_files = sorted((RAW_DATA_DIR / "isarcasm").glob("*.csv"))

    idx_offset = 1000  # Offset to avoid ID collisions with MUStARD

    if hf_path.exists():
        print(f"[iSarcasm] Loading from HF download {hf_path}")
        with open(hf_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                entry = json.loads(line.strip())
                text = entry.get("tweet", entry.get("text", ""))
                label_val = entry.get("label", entry.get("sarcasm", 0))
                is_sarcastic = bool(int(label_val)) if label_val is not None else False

                item = {
                    "id": f"sarc_{idx_offset + idx:04d}",
                    "context": "Tweet (no prior dialogue context available).",
                    "speaker": "twitter-user",
                    "utterance": str(text),
                    "literal_meaning": f"The literal content of the tweet.",
                    "intended_meaning": (
                        "The tweet is sarcastic — the intended meaning opposes or mocks the literal reading."
                        if is_sarcastic
                        else "The tweet is sincere."
                    ),
                    "pragmatic_category": "sarcasm",
                    "label": "sarcastic" if is_sarcastic else "sincere",
                    "answer_options": ["sincere", "sarcastic"],
                    "difficulty": "medium",
                    "requires_context": False,
                    "source": "iSarcasm",
                    "license": "research-use",
                    "split": "train",
                    "annotator_notes": "",
                }
                items.append(item)

    elif csv_files:
        for csv_path in csv_files:
            print(f"[iSarcasm] Loading from {csv_path}")
            with open(csv_path, "r", encoding="utf-8", errors="replace") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header is None:
                    continue

                # Try to find the text and label columns
                text_col = None
                label_col = None
                for i, h in enumerate(header):
                    h_lower = h.lower().strip()
                    if h_lower in ("tweet", "text", "tweet_text"):
                        text_col = i
                    elif h_lower in ("label", "sarcasm_label", "sarcastic", "sarcasm"):
                        label_col = i

                if text_col is None or label_col is None:
                    print(f"[iSarcasm] Skipping {csv_path.name}: tweet text and label columns required (IDs are not text).")
                    continue

                for idx, row in enumerate(reader):
                    if len(row) <= max(text_col, label_col):
                        continue

                    text = row[text_col].strip()
                    if not text:
                        continue

                    try:
                        raw_label = row[label_col].strip().lower()
                        is_sarcastic = {"1": True, "0": False, "sarcastic": True, "not_sarcastic": False}[raw_label]
                    except (KeyError, IndexError):
                        raise ValueError(f"Unknown iSarcasm label in {csv_path.name}: {row[label_col]!r}")

                    item = {
                        "id": f"sarc_{idx_offset + len(items):04d}",
                        "context": "Tweet (no prior dialogue context available).",
                        "speaker": "twitter-user",
                        "utterance": text,
                        "literal_meaning": "The literal content of the tweet.",
                        "intended_meaning": (
                            "The tweet is sarcastic."
                            if is_sarcastic
                            else "The tweet is sincere."
                        ),
                        "pragmatic_category": "sarcasm",
                        "label": "sarcastic" if is_sarcastic else "sincere",
                        "answer_options": ["sincere", "sarcastic"],
                        "difficulty": "medium",
                        "requires_context": False,
                        "source": "iSarcasm",
                        "license": "research-use",
                        "split": "train",
                        "annotator_notes": f"Source file: {csv_path.name}",
                    }
                    items.append(item)
    else:
        print(f"[iSarcasm] ✗ No source files found. Run download.py first.")
        return []

    print(f"[iSarcasm] ✓ Normalized {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Circa normalization
# ---------------------------------------------------------------------------


def normalize_circa() -> list[dict[str, Any]]:
    """Normalize Circa dataset to unified schema.

    Circa provides indirect answers to polar (yes/no) questions, with labels
    indicating whether the answer implies yes, no, or is ambiguous.
    The TSV has columns: context/scenario, question, answer, label, etc.
    """
    items: list[dict[str, Any]] = []
    hf_path = RAW_DATA_DIR / "circa" / "circa_hf.jsonl"
    tsv_path = RAW_DATA_DIR / "circa" / "circa-data.tsv"

    # Circa label mapping — map Circa's specific labels to our schema
    CIRCA_LABEL_MAP = {
        "Yes, subject did.": "yes",
        "Probably yes / sometimes yes": "yes",
        "Yes, subject to some conditions": "uncertain",
        "Yes, subject did. [yes]": "yes",
        "No, subject did not.": "no",
        "Probably no": "no",
        "No, subject did not. [no]": "no",
        "In the middle, neither yes nor no": "uncertain",
        "I am not sure how X will interpret Y's answer": "uncertain",
        "Other": "uncertain",
        "NA": "uncertain",
        "Yes": "yes",
        "No": "no",
    }

    if hf_path.exists():
        print(f"[Circa] Loading from HF download {hf_path}")
        with open(hf_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(f):
                entry = json.loads(line.strip())

                context = entry.get("context", entry.get("scenario", ""))
                question = entry.get("question", entry.get("question_text", ""))
                answer = entry.get("answer", entry.get("answer_text", ""))
                raw_label = str(entry.get("goldstandard1", entry.get("label", "uncertain")))

                label = CIRCA_LABEL_MAP.get(raw_label, "uncertain")

                item = {
                    "id": f"impl_{idx + 1:04d}",
                    "context": f"Scenario: {context} A asks: \"{question}\"" if context else f"A asks: \"{question}\"",
                    "speaker": "B",
                    "utterance": str(answer),
                    "literal_meaning": f"B literally says: \"{answer}\"",
                    "intended_meaning": (
                        f"B implies 'yes' to A's question."
                        if label == "yes"
                        else f"B implies 'no' to A's question."
                        if label == "no"
                        else "B's answer is ambiguous or non-committal."
                    ),
                    "pragmatic_category": "implicature",
                    "label": label,
                    "answer_options": ["yes", "no", "uncertain"],
                    "difficulty": "medium" if label in ("yes", "no") else "hard",
                    "requires_context": True,
                    "source": "Circa",
                    "license": "research-use",
                    "split": "train",
                    "annotator_notes": f"Original label: {raw_label}",
                }
                items.append(item)

    elif tsv_path.exists():
        print(f"[Circa] Loading from {tsv_path}")
        with open(tsv_path, "r", encoding="utf-8", errors="replace") as f:
            # Circa is literal TSV: quotes in dialogue are ordinary text.
            reader = csv.reader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
            header = next(reader, None)

            if header is None:
                print("[Circa] ✗ Empty TSV file")
                return []

            # Map column names to indices
            col_map = {h.strip().lower(): i for i, h in enumerate(header)}

            for idx, row in enumerate(reader):
                if len(row) != len(header):
                    raise ValueError(f"Malformed Circa row {idx + 2}: {len(row)} columns")

                # Try to extract columns by name, falling back to position
                context = row[col_map.get("context", 1)].strip() if "context" in col_map else row[1].strip()
                question = row[col_map.get("question-x", 2)].strip() if "question-x" in col_map else row[2].strip()
                answer = row[col_map.get("answer-y", 3)].strip() if "answer-y" in col_map else row[3].strip()

                # Try to get the goldstandard label
                raw_label = ""
                for key in ["goldstandard1", "goldstandard2", "label"]:
                    if key in col_map and col_map[key] < len(row):
                        raw_label = row[col_map[key]].strip()
                        if raw_label:
                            break
                if not raw_label and len(row) > 4:
                    raw_label = row[4].strip()

                label = CIRCA_LABEL_MAP.get(raw_label, "uncertain")

                if not answer:
                    continue

                item = {
                    "id": f"impl_{idx + 1:04d}",
                    "context": f"Scenario: {context} A asks: \"{question}\"" if context else f"A asks: \"{question}\"",
                    "speaker": "B",
                    "utterance": answer,
                    "literal_meaning": f"B literally says: \"{answer}\"",
                    "intended_meaning": (
                        "B implies 'yes' to A's question."
                        if label == "yes"
                        else "B implies 'no' to A's question."
                        if label == "no"
                        else "B's answer is ambiguous or non-committal."
                    ),
                    "pragmatic_category": "implicature",
                    "label": label,
                    "answer_options": ["yes", "no", "uncertain"],
                    "difficulty": "medium" if label in ("yes", "no") else "hard",
                    "requires_context": True,
                    "source": "Circa",
                    "license": "research-use",
                    "split": "train",
                    "annotator_notes": f"Original label: {raw_label}",
                }
                items.append(item)
    else:
        print("[Circa] ✗ No source files found. Run download.py first.")
        return []

    print(f"[Circa] ✓ Normalized {len(items)} items")
    return items


# ---------------------------------------------------------------------------
# Indirect speech acts (pass-through — already in schema from download.py)
# ---------------------------------------------------------------------------


def normalize_indirect() -> list[dict[str, Any]]:
    """Load pre-formatted indirect speech act dataset examples.

    These are already in unified schema format from curate_indirect.py / download.py.
    This function reads them and re-validates.
    """
    dataset_path = RAW_DATA_DIR / "indirect_speech_acts" / "indirect_dataset.jsonl"
    seed_path = RAW_DATA_DIR / "indirect_speech_acts" / "indirect_seed_examples.jsonl"

    source_path = dataset_path if dataset_path.exists() else seed_path

    if not source_path.exists():
        print("[Indirect] ✗ No dataset or seed file found. Run curate_indirect.py first.")
        return []

    items: list[dict[str, Any]] = []
    with open(source_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = json.loads(line.strip())
                items.append(item)

    print(f"[Indirect] ✓ Loaded {len(items)} items from {source_path.name}")
    return items


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------


def write_processed(items: list[dict[str, Any]], source_name: str) -> Path:
    """Write normalized items to a JSONL file in data/processed/."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DIR / f"{source_name}.jsonl"
    for item in items:
        errors = validate_item(item)
        if errors:
            raise ValueError(f"Invalid {source_name} item {item.get('id')}: {errors}")

    # Validate all items before writing
    valid_count = 0
    invalid_count = 0

    with open(output_path, "w", encoding="utf-8") as f:
        for item in items:
            errors = validate_item(item)
            if errors:
                invalid_count += 1
                print(f"  ⚠ Validation warning for {item.get('id', '?')}: {errors[0]}")
            else:
                valid_count += 1
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"  → Wrote {len(items)} items to {output_path} ({valid_count} valid, {invalid_count} with warnings)")
    return output_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Normalize raw datasets into the unified benchmark schema."
    )
    parser.add_argument("--all", action="store_true", help="Normalize all sources")
    parser.add_argument("--mustard", action="store_true", help="Normalize MUStARD")
    parser.add_argument("--isarcasm", action="store_true", help="Normalize iSarcasm")
    parser.add_argument("--circa", action="store_true", help="Normalize Circa")
    parser.add_argument("--indirect", action="store_true", help="Normalize indirect speech acts")

    args = parser.parse_args()

    if not any([args.all, args.mustard, args.isarcasm, args.circa, args.indirect]):
        args.all = True

    all_items: list[dict[str, Any]] = []

    if args.all or args.mustard:
        items = normalize_mustard()
        if items:
            write_processed(items, "mustard")
            all_items.extend(items)
        print()

    if args.all or args.isarcasm:
        items = normalize_isarcasm()
        if items:
            write_processed(items, "isarcasm")
            all_items.extend(items)
        print()

    if args.all or args.circa:
        items = normalize_circa()
        if items:
            write_processed(items, "circa")
            all_items.extend(items)
        print()

    if args.all or args.indirect:
        items = normalize_indirect()
        if items:
            write_processed(items, "indirect_speech_acts")
            all_items.extend(items)
        print()

    # Write combined file
    if all_items:
        write_processed(all_items, "all_combined")
        print(f"\n{'=' * 60}")
        print(f"TOTAL: {len(all_items)} items across all sources")

        # Category distribution
        cats: dict[str, int] = {}
        for item in all_items:
            cat = item.get("pragmatic_category", "unknown")
            cats[cat] = cats.get(cat, 0) + 1
        print("Category distribution:")
        for cat, count in sorted(cats.items()):
            print(f"  {cat}: {count}")
        print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
