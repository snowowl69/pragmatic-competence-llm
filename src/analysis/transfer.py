"""
Cross-category transfer matrix builder.

Builds the transfer matrix from Section 15 of the execution guide:
  - Rows: training condition (sarcasm-only, implicature-only, indirect-only, joint)
  - Columns: test category (sarcasm, implicature, indirect speech acts)
  - Cells: accuracy on that test category

This matrix is the project's core research artifact (RQ3).

Usage:
    python src/analysis/transfer.py --results-dir results/ --output results/transfer_matrix.csv
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def build_transfer_matrix(
    results_dir: str | Path,
    baseline_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build the cross-category transfer matrix from evaluation results.

    Expects results directories structured as:
        results/
        ├── baseline/          # Zero-shot baseline results
        ├── post_ft/
        │   ├── joint/         # Joint fine-tuning results
        │   ├── sarcasm/       # Sarcasm-only fine-tuning results
        │   ├── implicature/   # Implicature-only fine-tuning results
        │   └── indirect/      # Indirect-only fine-tuning results

    Returns:
        {
            "matrix": {train_condition: {test_category: accuracy}},
            "baseline": {test_category: accuracy},
            "transfer_gains": {train_condition: {test_category: gain_over_baseline}},
        }
    """
    results_dir = Path(results_dir)
    categories = ["sarcasm", "implicature", "indirect_speech_act"]
    train_conditions = ["sarcasm", "implicature", "indirect", "joint"]

    # Load and compute per-category accuracy for each condition
    matrix: dict[str, dict[str, float]] = {}
    baseline: dict[str, float] = {}

    # Load baseline results
    if baseline_dir:
        baseline_path = Path(baseline_dir)
    else:
        baseline_path = results_dir / "baseline"

    if baseline_path.exists():
        baseline_results = _load_results_from_dir(baseline_path)
        baseline = _compute_per_category_accuracy(baseline_results, categories)

    # Load post-FT results for each training condition
    post_ft_dir = results_dir / "post_ft"
    for condition in train_conditions:
        condition_dir = post_ft_dir / condition
        if condition_dir.exists():
            condition_results = _load_results_from_dir(condition_dir)
            if baseline:
                def protocol(row):
                    return tuple(row.get(k) for k in ('model_id', 'setting', 'prompt_template', 'context_mode', 'run_index', 'benchmark_sha256', 'inference_method'))
                if {protocol(r) for r in baseline_results} != {protocol(r) for r in condition_results}:
                    raise ValueError('Baseline and adapter evaluation protocols differ')
                gold = lambda rows: {r['item_id']: (r['pragmatic_category'], r['gold_label']) for r in rows}
                if gold(baseline_results) != gold(condition_results):
                    raise ValueError('Baseline and adapter test IDs or gold labels differ')
            matrix[condition] = _compute_per_category_accuracy(condition_results, categories)

    # Compute transfer gains over baseline
    transfer_gains: dict[str, dict[str, float]] = {}
    for condition, cat_accs in matrix.items():
        transfer_gains[condition] = {}
        for cat, acc in cat_accs.items():
            baseline_acc = baseline.get(cat)
            transfer_gains[condition][cat] = acc - baseline_acc if acc is not None and baseline_acc is not None else None

    return {
        "matrix": matrix,
        "baseline": baseline,
        "transfer_gains": transfer_gains,
        "categories": categories,
        "conditions": list(matrix.keys()),
    }


def _load_results_from_dir(directory: Path) -> list[dict[str, Any]]:
    """Load all JSONL result files from a directory."""
    results: list[dict[str, Any]] = []
    for jsonl_file in directory.glob("*.jsonl"):
        with open(jsonl_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    results.append(json.loads(line.strip()))
    return results


def _compute_per_category_accuracy(
    results: list[dict[str, Any]],
    categories: list[str],
) -> dict[str, float]:
    """Compute accuracy per pragmatic category from result records."""
    correct: dict[str, int] = defaultdict(int)
    total: dict[str, int] = defaultdict(int)

    seen = set()
    protocols = set()
    for r in results:
        key = r.get('item_id')
        if not key or key in seen: raise ValueError('Missing or repeated test item ID')
        seen.add(key)
        protocols.add(tuple(r.get(k) for k in ('model', 'setting', 'prompt_template', 'context_mode', 'run_index', 'benchmark_sha256', 'inference_method')))
        if len(protocols)>1: raise ValueError('Mixed evaluation protocols must not be pooled')
        cat = r.get("pragmatic_category", "unknown")
        if cat in categories:
            total[cat] += 1
            if r.get("gold_label") is not None and r.get("predicted_label") == r["gold_label"]:
                correct[cat] += 1

    return {
        cat: correct[cat] / total[cat] if total[cat] > 0 else None
        for cat in categories
    }


# ---------------------------------------------------------------------------
# Output and display
# ---------------------------------------------------------------------------


def format_score(value):
    return 'Not measured' if value is None else f'{value:.4f}'


def print_transfer_matrix(data: dict[str, Any]) -> None:
    """Print the transfer matrix as a formatted table."""
    categories = data["categories"]
    cat_short = {"sarcasm": "Sarc", "implicature": "Impl", "indirect_speech_act": "Indirect"}

    print(f"\n{'=' * 70}")
    print("CROSS-CATEGORY TRANSFER MATRIX")
    print(f"{'=' * 70}")

    # Header
    header = f"{'Trained on':<20}"
    for cat in categories:
        header += f" {cat_short.get(cat, cat):>12}"
    print(header)
    print("-" * 60)

    # Baseline row
    if data["baseline"]:
        row = f"{'Baseline (0-shot)':<20}"
        for cat in categories:
            acc = data["baseline"].get(cat)
            row += f" {format_score(acc):>11}"
        print(row)
        print("-" * 60)

    # Fine-tuned rows
    for condition in data.get("conditions", []):
        cat_accs = data["matrix"].get(condition, {})
        row = f"{condition:<20}"
        for cat in categories:
            acc = cat_accs.get(cat)
            row += f" {format_score(acc):>11}"
        print(row)

    # Transfer gains
    if data["transfer_gains"]:
        print("-" * 60)
        print("\nTransfer gains over baseline:")
        for condition, gains in data["transfer_gains"].items():
            row = f"  {condition:<18}"
            for cat in categories:
                gain = gains.get(cat)
                row += f" {format_score(gain):>11}"
            print(row)

    print(f"{'=' * 70}\n")


def save_transfer_matrix_csv(
    data: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Save the transfer matrix as a CSV file."""
    import csv

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = data["categories"]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["trained_on"] + categories)

        # Baseline
        if data["baseline"]:
            writer.writerow(["baseline"] + [format_score(data['baseline'].get(c)) for c in categories])

        # Fine-tuned conditions
        for condition in data.get("conditions", []):
            accs = data["matrix"].get(condition, {})
            writer.writerow([condition] + [format_score(accs.get(c)) for c in categories])

    print(f"Transfer matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the cross-category transfer matrix from evaluation results."
    )
    parser.add_argument("--results-dir", type=str, default="results/",
                        help="Root results directory")
    parser.add_argument("--baseline-dir", type=str, default=None,
                        help="Baseline results directory (default: results/baseline)")
    parser.add_argument("--output", type=str, default="results/transfer_matrix.csv",
                        help="Output CSV path")

    args = parser.parse_args()

    data = build_transfer_matrix(args.results_dir, args.baseline_dir)
    print_transfer_matrix(data)
    save_transfer_matrix_csv(data, args.output)


if __name__ == "__main__":
    main()
