"""
Unified JSON schema definition and validation for the pragmatic competence benchmark.

This module defines the canonical schema from Section 6 of the execution guide,
and provides validation utilities used by the normalization pipeline (normalize.py),
the splitting pipeline (split.py), and the test suite (tests/test_schema.py).

Schema fields:
    id                 — Unique identifier (e.g., "sarc_0001", "impl_0042")
    context            — Situational/dialogue context preceding the utterance
    speaker            — Speaker role or relationship (e.g., "friend", "manager")
    utterance          — The target utterance to be interpreted
    literal_meaning    — What the utterance literally says
    intended_meaning   — What the speaker actually means (gold reference)
    pragmatic_category — One of: sarcasm, implicature, indirect_speech_act
    label              — Category-specific gold label
    answer_options     — List of valid label choices for this item
    difficulty         — easy / medium / hard
    requires_context   — Whether context is needed to resolve the item
    source             — Origin dataset or "hand-curated" / "control-item"
    license            — License of the source data
    split              — train / val / test
    annotator_notes    — Free-text annotation notes (may be empty)
"""

import json
from pathlib import Path
from typing import Any

from jsonschema import Draft7Validator, ValidationError


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

PRAGMATIC_CATEGORIES = {"sarcasm", "implicature", "indirect_speech_act"}

VALID_LABELS = {
    "sarcasm": {"sincere", "sarcastic"},
    "implicature": {"yes", "no", "uncertain", "ambiguous"},
    "indirect_speech_act": {"statement", "request", "question", "suggestion", "complaint"},
}

VALID_DIFFICULTIES = {"easy", "medium", "hard"}
VALID_SPLITS = {"train", "val", "test"}

# JSON Schema (Draft 7) for a single benchmark item
BENCHMARK_ITEM_SCHEMA: dict[str, Any] = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "title": "PragmaticBenchmarkItem",
    "description": "A single item in the pragmatic competence benchmark.",
    "type": "object",
    "required": [
        "id",
        "context",
        "speaker",
        "utterance",
        "literal_meaning",
        "intended_meaning",
        "pragmatic_category",
        "label",
        "answer_options",
        "difficulty",
        "requires_context",
        "source",
        "split",
    ],
    "properties": {
        "id": {
            "type": "string",
            "pattern": "^(sarc|impl|ind|ctrl|hard)_[0-9]{4,}$",
            "description": "Unique item ID with category prefix and numeric suffix.",
        },
        "context": {
            "type": "string",
            "minLength": 1,
            "description": "Situational or dialogue context.",
        },
        "speaker": {
            "type": "string",
            "minLength": 1,
            "description": "Speaker role or relationship.",
        },
        "utterance": {
            "type": "string",
            "minLength": 1,
            "description": "The target utterance to interpret.",
        },
        "literal_meaning": {
            "type": "string",
            "minLength": 1,
            "description": "Literal meaning of the utterance.",
        },
        "intended_meaning": {
            "type": "string",
            "minLength": 1,
            "description": "Gold pragmatic interpretation.",
        },
        "pragmatic_category": {
            "type": "string",
            "enum": sorted(PRAGMATIC_CATEGORIES),
            "description": "Pragmatic phenomenon category.",
        },
        "label": {
            "type": "string",
            "description": "Category-specific gold label.",
        },
        "answer_options": {
            "type": "array",
            "items": {"type": "string"},
            "minItems": 2,
            "description": "Valid answer choices for multiple-choice evaluation.",
        },
        "difficulty": {
            "type": "string",
            "enum": sorted(VALID_DIFFICULTIES),
            "description": "Annotator-assessed difficulty.",
        },
        "requires_context": {
            "type": "boolean",
            "description": "Whether context is needed to resolve the item.",
        },
        "source": {
            "type": "string",
            "minLength": 1,
            "description": "Origin dataset name.",
        },
        "license": {
            "type": "string",
            "description": "License of the source data.",
        },
        "split": {
            "type": "string",
            "enum": sorted(VALID_SPLITS),
            "description": "Dataset split assignment.",
        },
        "annotator_notes": {
            "type": "string",
            "description": "Free-text annotation notes.",
        },
    },
    "additionalProperties": False,
}

_validator = Draft7Validator(BENCHMARK_ITEM_SCHEMA)


# ---------------------------------------------------------------------------
# Validation functions
# ---------------------------------------------------------------------------


def validate_item(item: dict[str, Any]) -> list[str]:
    """Validate a single benchmark item against the schema.

    Returns a list of error messages (empty if valid).
    Also performs cross-field checks that JSON Schema alone cannot express
    (e.g., label must be valid for the item's pragmatic_category).
    """
    errors: list[str] = []

    # JSON Schema structural validation
    for error in sorted(_validator.iter_errors(item), key=lambda e: list(e.path)):
        path = ".".join(str(p) for p in error.absolute_path) or "(root)"
        errors.append(f"[schema] {path}: {error.message}")

    # Cross-field checks (only if structural validation passed enough)
    category = item.get("pragmatic_category")
    label = item.get("label")
    answer_options = item.get("answer_options")

    if category in VALID_LABELS and label is not None:
        # Allow "ambiguous" as a special cross-category label for hard items
        valid_for_category = VALID_LABELS[category] | {"ambiguous"}
        if label not in valid_for_category:
            errors.append(
                f"[cross-field] label '{label}' is not valid for "
                f"category '{category}'. Valid: {sorted(valid_for_category)}"
            )

    if answer_options is not None and label is not None:
        # Label should be one of the answer options (or "ambiguous" for flagged items)
        if label not in answer_options and label != "ambiguous":
            errors.append(
                f"[cross-field] label '{label}' is not in answer_options {answer_options}"
            )

    return errors


def validate_item_strict(item: dict[str, Any]) -> None:
    """Validate a single item; raise ValidationError on the first problem."""
    errors = validate_item(item)
    if errors:
        raise ValidationError(
            f"Validation failed for item '{item.get('id', '???')}': {errors[0]}",
            instance=item,
        )


def validate_jsonl_file(filepath: str | Path) -> dict[str, Any]:
    """Validate an entire JSONL benchmark file.

    Returns:
        {
            "total": int,
            "valid": int,
            "invalid": int,
            "errors": [{"line": int, "id": str, "errors": [str]}],
            "category_counts": {"sarcasm": int, ...},
            "split_counts": {"train": int, ...},
            "ids": set of all ids (for duplicate checking),
        }
    """
    filepath = Path(filepath)
    results: dict[str, Any] = {
        "total": 0,
        "valid": 0,
        "invalid": 0,
        "errors": [],
        "category_counts": {},
        "split_counts": {},
        "ids": set(),
    }

    with filepath.open("r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            results["total"] += 1

            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                results["invalid"] += 1
                results["errors"].append({
                    "line": line_num,
                    "id": "PARSE_ERROR",
                    "errors": [f"JSON parse error: {e}"],
                })
                continue

            item_errors = validate_item(item)
            item_id = item.get("id", f"line_{line_num}")

            if item_errors:
                results["invalid"] += 1
                results["errors"].append({
                    "line": line_num,
                    "id": item_id,
                    "errors": item_errors,
                })
            else:
                results["valid"] += 1

            # Track counts regardless of validity
            cat = item.get("pragmatic_category", "unknown")
            results["category_counts"][cat] = results["category_counts"].get(cat, 0) + 1

            split = item.get("split", "unknown")
            results["split_counts"][split] = results["split_counts"].get(split, 0) + 1

            # Track IDs for duplicate detection
            if item_id in results["ids"]:
                results["errors"].append({
                    "line": line_num,
                    "id": item_id,
                    "errors": [f"Duplicate ID: '{item_id}'"],
                })
            results["ids"].add(item_id)

    return results


def print_validation_report(results: dict[str, Any]) -> None:
    """Print a human-readable validation report."""
    print(f"\n{'=' * 60}")
    print(f"BENCHMARK VALIDATION REPORT")
    print(f"{'=' * 60}")
    print(f"Total items:   {results['total']}")
    print(f"Valid:         {results['valid']}")
    print(f"Invalid:       {results['invalid']}")
    print()

    if results["category_counts"]:
        print("Category distribution:")
        for cat, count in sorted(results["category_counts"].items()):
            print(f"  {cat}: {count}")
        print()

    if results["split_counts"]:
        print("Split distribution:")
        for split, count in sorted(results["split_counts"].items()):
            print(f"  {split}: {count}")
        print()

    if results["errors"]:
        print(f"ERRORS ({len(results['errors'])}):")
        for err in results["errors"][:20]:  # Show first 20
            print(f"  Line {err['line']} [{err['id']}]:")
            for e in err["errors"]:
                print(f"    - {e}")
        if len(results["errors"]) > 20:
            print(f"  ... and {len(results['errors']) - 20} more errors")
    else:
        print("✓ No errors found.")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate a pragmatic benchmark JSONL file against the unified schema."
    )
    parser.add_argument(
        "filepath",
        type=str,
        help="Path to the JSONL file to validate.",
    )
    args = parser.parse_args()

    results = validate_jsonl_file(args.filepath)
    print_validation_report(results)

    # Exit with error code if any items are invalid
    exit(0 if results["invalid"] == 0 else 1)
