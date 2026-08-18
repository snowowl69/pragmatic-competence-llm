"""
Failure taxonomy auto-tagger and manual inspection support.

Implements the 7-category failure taxonomy from Section 12 of the
execution guide, with automatic tagging where possible and support
for manual annotation of a stratified error sample.

Auto-detectable failure types:
  - Context under-use: compare accuracy with/without context
  - Lexical-shortcut: compare accuracy on marker items vs. controls

Manual tagging support:
  - Exports stratified error samples for manual inspection
  - Produces failure-type distribution charts
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Failure taxonomy definitions
# ---------------------------------------------------------------------------

FAILURE_TYPES = {
    "literal_bias": {
        "description": "Model answers as if only literal semantics matters.",
        "auto_detectable": False,
        "manual_signal": "Model's prediction matches the literal reading, not the pragmatic one.",
    },
    "context_underuse": {
        "description": "Model ignores available disambiguating context.",
        "auto_detectable": True,
        "detection_method": "Compare prediction with/without context — if same wrong answer in both conditions.",
    },
    "world_knowledge_gap": {
        "description": "Needs outside knowledge (cultural norm, domain fact) the model lacks.",
        "auto_detectable": False,
        "manual_signal": "Correct answer requires knowledge not present in context or utterance.",
    },
    "lexical_shortcut": {
        "description": "Relies on surface trigger words rather than understanding.",
        "auto_detectable": True,
        "detection_method": "Compare accuracy on marker-heavy items vs. matched control items.",
    },
    "negation_scope_error": {
        "description": "Mishandles negation or scope within the utterance.",
        "auto_detectable": False,  # Partial — can flag items containing negation
        "manual_signal": "Utterance contains negation and model misinterprets scope.",
    },
    "ambiguity": {
        "description": "Item is genuinely ambiguous even to human annotators.",
        "auto_detectable": False,
        "detection_method": "Pre-flagged during annotation (label=ambiguous).",
    },
    "reasoning_chain_failure": {
        "description": "Model shows correct reasoning steps but wrong final label (or vice versa).",
        "auto_detectable": False,
        "manual_signal": "Inspect raw model output for reasoning quality vs. final answer.",
    },
}


# ---------------------------------------------------------------------------
# Auto-tagging
# ---------------------------------------------------------------------------


def tag_context_underuse(
    results_with_context: list[dict[str, Any]],
    results_without_context: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Tag errors where context removal didn't change the (wrong) prediction.

    If the model gives the same wrong answer with and without context,
    it wasn't using context — classic context under-use.
    """
    # Index results by item_id
    no_ctx = {r["item_id"]: r for r in results_without_context}

    tagged: list[dict[str, Any]] = []
    for r in results_with_context:
        if r["correct"]:
            continue  # Only tag errors

        item_id = r["item_id"]
        if item_id in no_ctx:
            no_ctx_result = no_ctx[item_id]
            if (not no_ctx_result["correct"] and
                    r["predicted_label"] == no_ctx_result["predicted_label"]):
                tagged.append({
                    "item_id": item_id,
                    "failure_type": "context_underuse",
                    "evidence": f"Same wrong prediction ({r['predicted_label']}) with and without context.",
                    "predicted_label": r["predicted_label"],
                    "gold_label": r["gold_label"],
                    "category": r.get("pragmatic_category", "unknown"),
                })

    return tagged


def tag_lexical_shortcut(
    results: list[dict[str, Any]],
    benchmark_items: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Detect lexical-shortcut reliance by comparing marker vs. control item accuracy.

    Items with source="control-item" are designed to have the same surface markers
    as typical sarcasm/implicature triggers but with the opposite label.
    A large accuracy gap (high on marker items, low on controls) indicates shortcut reliance.
    """
    marker_correct = 0
    marker_total = 0
    control_correct = 0
    control_total = 0

    for r in results:
        item_id = r["item_id"]
        if item_id in benchmark_items:
            source = benchmark_items[item_id].get("source", "")
            if source == "control-item":
                control_total += 1
                if r["correct"]:
                    control_correct += 1
            else:
                marker_total += 1
                if r["correct"]:
                    marker_correct += 1

    marker_acc = marker_correct / marker_total if marker_total > 0 else 0
    control_acc = control_correct / control_total if control_total > 0 else 0
    gap = marker_acc - control_acc

    return {
        "marker_accuracy": marker_acc,
        "marker_count": marker_total,
        "control_accuracy": control_acc,
        "control_count": control_total,
        "accuracy_gap": gap,
        "shortcut_signal": gap > 0.15,  # >15% gap suggests shortcut reliance
    }


# ---------------------------------------------------------------------------
# Manual tagging support
# ---------------------------------------------------------------------------


def export_error_sample(
    results: list[dict[str, Any]],
    benchmark_items: dict[str, dict[str, Any]],
    n_per_category: int = 15,
    output_path: str | Path = "results/failure_analysis/error_sample.jsonl",
) -> int:
    """Export a stratified error sample for manual failure-type tagging.

    Samples up to n_per_category errors per pragmatic category,
    enriched with the original benchmark item data and raw model output.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Group errors by category
    errors_by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in results:
        if not r["correct"]:
            errors_by_cat[r.get("pragmatic_category", "unknown")].append(r)

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for category, errors in sorted(errors_by_cat.items()):
            # Take up to n_per_category
            sample = errors[:n_per_category]

            for error in sample:
                item_id = error["item_id"]
                benchmark = benchmark_items.get(item_id, {})

                tagged_item = {
                    "item_id": item_id,
                    "pragmatic_category": category,
                    "context": benchmark.get("context", ""),
                    "utterance": benchmark.get("utterance", ""),
                    "gold_label": error["gold_label"],
                    "predicted_label": error["predicted_label"],
                    "raw_output": error.get("raw_output", ""),
                    "model": error.get("model", ""),
                    "difficulty": benchmark.get("difficulty", ""),
                    "source": benchmark.get("source", ""),
                    # Fields for manual tagging:
                    "failure_type": "",  # TO BE FILLED BY ANNOTATOR
                    "failure_notes": "",  # TO BE FILLED BY ANNOTATOR
                }

                f.write(json.dumps(tagged_item, ensure_ascii=False) + "\n")
                count += 1

    print(f"Exported {count} errors for manual tagging → {output_path}")
    return count


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------


def aggregate_failure_types(
    tagged_errors: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Aggregate failure types by pragmatic category.

    Returns: {category: {failure_type: count}}
    """
    agg: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for err in tagged_errors:
        cat = err.get("pragmatic_category", err.get("category", "unknown"))
        ftype = err.get("failure_type", "untagged")
        if ftype:
            agg[cat][ftype] += 1

    return dict(agg)


def print_failure_report(
    aggregated: dict[str, dict[str, int]],
) -> None:
    """Print a failure-type distribution report."""
    print(f"\n{'=' * 70}")
    print("FAILURE TAXONOMY DISTRIBUTION")
    print(f"{'=' * 70}")

    all_types = set()
    for cat_data in aggregated.values():
        all_types.update(cat_data.keys())

    # Header
    header = f"{'Type':<25}"
    for cat in sorted(aggregated.keys()):
        header += f" {cat:>15}"
    header += f" {'Total':>10}"
    print(header)
    print("-" * len(header))

    # Rows
    for ftype in sorted(all_types):
        row = f"{ftype:<25}"
        total = 0
        for cat in sorted(aggregated.keys()):
            count = aggregated[cat].get(ftype, 0)
            row += f" {count:>15}"
            total += count
        row += f" {total:>10}"
        print(row)

    print(f"{'=' * 70}\n")
