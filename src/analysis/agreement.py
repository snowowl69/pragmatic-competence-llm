"""
Inter-annotator agreement computation (Cohen's κ).

Used during Phase 3 (Annotation & QC) to measure annotation consistency
on the double-annotated sample.

Usage:
    python src/analysis/agreement.py --file data/processed/double_annotated.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import Any

from sklearn.metrics import cohen_kappa_score


def compute_agreement(
    annotations_a: list[str],
    annotations_b: list[str],
) -> dict[str, Any]:
    """Compute Cohen's κ and percentage agreement.

    Args:
        annotations_a: Labels from annotator A
        annotations_b: Labels from annotator B

    Returns:
        Dictionary with kappa, agreement percentage, and interpretation
    """
    assert len(annotations_a) == len(annotations_b), "Annotation lists must have equal length"

    kappa = cohen_kappa_score(annotations_a, annotations_b)
    n = len(annotations_a)
    agree = sum(1 for a, b in zip(annotations_a, annotations_b) if a == b)
    pct_agreement = agree / n if n > 0 else 0

    # Standard interpretation thresholds (Landis & Koch, 1977)
    if kappa < 0:
        interpretation = "Less than chance agreement"
    elif kappa < 0.20:
        interpretation = "Slight agreement"
    elif kappa < 0.40:
        interpretation = "Fair agreement"
    elif kappa < 0.60:
        interpretation = "Moderate agreement"
    elif kappa < 0.80:
        interpretation = "Substantial agreement"
    else:
        interpretation = "Almost perfect agreement"

    # Project threshold check (κ ≥ 0.6 per execution guide Section 7)
    meets_threshold = kappa >= 0.6

    return {
        "cohens_kappa": float(kappa),
        "percentage_agreement": float(pct_agreement),
        "n_items": n,
        "n_agree": agree,
        "n_disagree": n - agree,
        "interpretation": interpretation,
        "meets_project_threshold": meets_threshold,
    }


def compute_per_category_agreement(
    items: list[dict[str, Any]],
    label_key_a: str = "label_annotator_a",
    label_key_b: str = "label_annotator_b",
    category_key: str = "pragmatic_category",
) -> dict[str, dict[str, Any]]:
    """Compute Cohen's κ separately per pragmatic category.

    Expects items with both annotator labels and a category field.
    """
    from collections import defaultdict

    by_category: dict[str, dict[str, list]] = defaultdict(lambda: {"a": [], "b": []})

    for item in items:
        cat = item.get(category_key, "unknown")
        by_category[cat]["a"].append(item[label_key_a])
        by_category[cat]["b"].append(item[label_key_b])

    results: dict[str, dict[str, Any]] = {}
    for cat, data in sorted(by_category.items()):
        results[cat] = compute_agreement(data["a"], data["b"])

    return results


def print_agreement_report(
    overall: dict[str, Any],
    per_category: dict[str, dict[str, Any]] | None = None,
) -> None:
    """Print a human-readable agreement report."""
    print(f"\n{'=' * 60}")
    print("INTER-ANNOTATOR AGREEMENT REPORT")
    print(f"{'=' * 60}")
    print(f"Overall Cohen's κ:      {overall['cohens_kappa']:.3f}")
    print(f"Percentage agreement:   {overall['percentage_agreement']:.1%}")
    print(f"Items:                  {overall['n_items']} ({overall['n_agree']} agree, {overall['n_disagree']} disagree)")
    print(f"Interpretation:         {overall['interpretation']}")
    print(f"Meets threshold (≥0.6): {'✓ YES' if overall['meets_project_threshold'] else '✗ NO — revisit guidelines'}")

    if per_category:
        print(f"\nPer-category breakdown:")
        for cat, data in per_category.items():
            status = "✓" if data["meets_project_threshold"] else "✗"
            print(f"  {cat}: κ={data['cohens_kappa']:.3f} ({data['percentage_agreement']:.1%} agreement) {status}")

    print(f"{'=' * 60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute inter-annotator agreement (Cohen's κ) on double-annotated items."
    )
    parser.add_argument("--file", type=str, required=True,
                        help="JSONL file with double-annotated items")
    parser.add_argument("--label-a", type=str, default="label_annotator_a",
                        help="Field name for annotator A's labels")
    parser.add_argument("--label-b", type=str, default="label_annotator_b",
                        help="Field name for annotator B's labels")

    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"✗ File not found: {filepath}")
        return

    items: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line.strip()))

    labels_a = [item[args.label_a] for item in items]
    labels_b = [item[args.label_b] for item in items]

    overall = compute_agreement(labels_a, labels_b)
    per_category = compute_per_category_agreement(
        items, args.label_a, args.label_b
    )

    print_agreement_report(overall, per_category)


if __name__ == "__main__":
    main()
