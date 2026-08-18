"""
Statistical significance testing — McNemar's test wrapper.

Provides convenience functions for running McNemar's test across
all key baseline-vs-fine-tuned comparison pairs, generating a
report with p-values and significance flags.

Usage:
    python src/analysis/significance.py \\
        --baseline results/baseline/ \\
        --finetuned results/post_ft/joint/ \\
        --output results/significance_results.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import load_results, mcnemars_test


def run_significance_tests(
    baseline_dir: str | Path,
    finetuned_dir: str | Path,
    categories: list[str] | None = None,
) -> dict[str, Any]:
    """Run McNemar's test comparing baseline vs. fine-tuned on each category.

    Args:
        baseline_dir: Directory with baseline result JSONL files
        finetuned_dir: Directory with fine-tuned result JSONL files
        categories: Optional filter — only test these categories

    Returns:
        Dictionary mapping category → McNemar's test results
    """
    if categories is None:
        categories = ["sarcasm", "implicature", "indirect_speech_act"]

    # Load all results
    baseline_results = _load_all_results(Path(baseline_dir))
    finetuned_results = _load_all_results(Path(finetuned_dir))

    # Index by item_id
    baseline_by_id = {r["item_id"]: r for r in baseline_results}
    finetuned_by_id = {r["item_id"]: r for r in finetuned_results}

    # Find common items
    common_ids = set(baseline_by_id.keys()) & set(finetuned_by_id.keys())
    if not common_ids:
        print("⚠ No common items found between baseline and fine-tuned results.")
        return {}

    # Run per-category tests
    results: dict[str, Any] = {}

    # Overall test (all categories combined)
    gold_all = [baseline_by_id[i]["gold_label"] for i in common_ids]
    pred_a_all = [baseline_by_id[i]["predicted_label"] for i in common_ids]
    pred_b_all = [finetuned_by_id[i]["predicted_label"] for i in common_ids]

    results["overall"] = mcnemars_test(gold_all, pred_a_all, pred_b_all)
    results["overall"]["n_items"] = len(common_ids)

    # Per-category tests
    for cat in categories:
        cat_ids = [
            i for i in common_ids
            if baseline_by_id[i].get("pragmatic_category") == cat
        ]

        if len(cat_ids) < 5:
            results[cat] = {"error": f"Too few items ({len(cat_ids)}) for significance test"}
            continue

        gold = [baseline_by_id[i]["gold_label"] for i in cat_ids]
        pred_a = [baseline_by_id[i]["predicted_label"] for i in cat_ids]
        pred_b = [finetuned_by_id[i]["predicted_label"] for i in cat_ids]

        results[cat] = mcnemars_test(gold, pred_a, pred_b)
        results[cat]["n_items"] = len(cat_ids)

    return results


def _load_all_results(directory: Path) -> list[dict[str, Any]]:
    """Load all JSONL results from a directory."""
    results: list[dict[str, Any]] = []
    for f in directory.glob("*.jsonl"):
        results.extend(load_results(f))
    return results


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def print_significance_report(results: dict[str, Any]) -> None:
    """Print a formatted significance test report."""
    print(f"\n{'=' * 80}")
    print("McNEMAR'S TEST — BASELINE vs. FINE-TUNED")
    print(f"{'=' * 80}")
    print(f"{'Category':<25} {'p-value':>10} {'Sig.(0.05)':>12} {'Sig.(0.01)':>12} {'Δ Accuracy':>12} {'Method':>20}")
    print("-" * 90)

    for cat, data in results.items():
        if "error" in data:
            print(f"{cat:<25} {data['error']}")
            continue

        p = data["p_value"]
        sig05 = "✓ YES" if data["significant_at_005"] else "✗ no"
        sig01 = "✓ YES" if data["significant_at_001"] else "✗ no"
        delta = data["model_b_accuracy"] - data["model_a_accuracy"]
        method = data["method"]

        print(f"{cat:<25} {p:>10.4f} {sig05:>12} {sig01:>12} {delta:>+11.3f} {method:>20}")

    print(f"{'=' * 80}")
    print("Model A = baseline, Model B = fine-tuned")
    print("Positive Δ Accuracy = fine-tuned is better\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run McNemar's significance tests on baseline vs. fine-tuned results."
    )
    parser.add_argument("--baseline", type=str, required=True,
                        help="Directory with baseline result files")
    parser.add_argument("--finetuned", type=str, required=True,
                        help="Directory with fine-tuned result files")
    parser.add_argument("--output", type=str, default="results/significance_results.json",
                        help="Output JSON file for test results")

    args = parser.parse_args()

    results = run_significance_tests(args.baseline, args.finetuned)
    print_significance_report(results)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
