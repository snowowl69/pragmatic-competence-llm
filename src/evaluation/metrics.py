"""
Evaluation metrics for the pragmatic competence benchmark.

Computes accuracy, macro-F1, per-category metrics, confusion matrices,
and McNemar's test for paired comparisons. This module is used by both
the evaluation harness (harness.py) and the analysis scripts.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------


def compute_metrics(
    gold_labels: list[str],
    predicted_labels: list[str],
    label_names: list[str] | None = None,
) -> dict[str, Any]:
    """Compute accuracy, macro-F1, and per-label metrics.

    Args:
        gold_labels: Ground truth labels
        predicted_labels: Model predictions
        label_names: Optional ordered list of label names for the confusion matrix

    Returns:
        Dictionary with accuracy, macro_f1, per_label metrics, and confusion matrix
    """
    if not gold_labels or len(gold_labels) != len(predicted_labels):
        raise ValueError("Metrics require nonempty, equally sized gold and prediction arrays")
    accuracy = accuracy_score(gold_labels, predicted_labels)
    macro_f1 = f1_score(gold_labels, predicted_labels, labels=label_names, average="macro", zero_division=0)

    report = classification_report(
        gold_labels,
        predicted_labels,
        labels=label_names,
        output_dict=True,
        zero_division=0,
    )

    cm = confusion_matrix(gold_labels, predicted_labels, labels=label_names)

    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "label_names": label_names or sorted(set(gold_labels) | set(predicted_labels)),
        "n_items": len(gold_labels),
    }


def compute_per_category_metrics(
    results: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute metrics separately for each pragmatic category.

    Args:
        results: List of result dicts with at least 'pragmatic_category',
                 'gold_label', and 'predicted_label' keys.

    Returns:
        Dictionary mapping category → metrics dict
    """
    by_category: dict[str, dict[str, list]] = defaultdict(lambda: {"gold": [], "pred": []})

    for r in results:
        cat = r.get("pragmatic_category", "unknown")
        by_category[cat]["gold"].append(r["gold_label"])
        by_category[cat]["pred"].append(r["predicted_label"])

    category_metrics: dict[str, dict[str, Any]] = {}
    for cat, data in sorted(by_category.items()):
        category_metrics[cat] = compute_metrics(data["gold"], data["pred"])

    return category_metrics


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemars_test(
    gold_labels: list[str],
    predictions_a: list[str],
    predictions_b: list[str],
) -> dict[str, Any]:
    """Run McNemar's test comparing two models' predictions on the same test set.

    Tests whether the two models make significantly different errors.
    This is the correct paired test for classification predictions on the
    same items (Section I of the project plan).

    Args:
        gold_labels: Ground truth labels
        predictions_a: Model A's predictions (e.g., baseline)
        predictions_b: Model B's predictions (e.g., fine-tuned)

    Returns:
        Dictionary with contingency table, test statistic, and p-value.
    """
    if not gold_labels or not len(gold_labels) == len(predictions_a) == len(predictions_b):
        raise ValueError("Paired tests require nonempty, equally sized arrays")

    # Build the 2×2 contingency table:
    #               Model B correct  | Model B wrong
    # Model A correct    n00         |    n01
    # Model A wrong      n10         |    n11
    n00 = n01 = n10 = n11 = 0

    for gold, pred_a, pred_b in zip(gold_labels, predictions_a, predictions_b):
        a_correct = pred_a == gold
        b_correct = pred_b == gold

        if a_correct and b_correct:
            n00 += 1
        elif a_correct and not b_correct:
            n01 += 1
        elif not a_correct and b_correct:
            n10 += 1
        else:
            n11 += 1

    # McNemar's test focuses on the discordant cells (n01, n10)
    contingency = [[n00, n01], [n10, n11]]

    # Use exact binomial test if discordant count is small (< 25),
    # chi-squared approximation otherwise
    discordant = n01 + n10
    if discordant == 0:
        p_value = 1.0
        statistic = 0.0
        method = "no_discordant_pairs"
    elif discordant < 25:
        # Exact binomial test: under H0, n10 ~ Binom(n01+n10, 0.5)
        result = scipy_stats.binomtest(n10, discordant, 0.5)
        p_value = result.pvalue
        statistic = float(n10)
        method = "exact_binomial"
    else:
        # Chi-squared approximation with continuity correction
        statistic = max(0, abs(n01 - n10) - 1) ** 2 / (n01 + n10)
        p_value = scipy_stats.chi2.sf(statistic, df=1)
        method = "chi_squared_corrected"

    return {
        "contingency_table": contingency,
        "n00_both_correct": n00,
        "n01_a_correct_b_wrong": n01,
        "n10_a_wrong_b_correct": n10,
        "n11_both_wrong": n11,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "method": method,
        "significant_at_005": bool(p_value < 0.05),
        "significant_at_001": bool(p_value < 0.01),
        "model_a_accuracy": (n00 + n01) / (n00 + n01 + n10 + n11),
        "model_b_accuracy": (n00 + n10) / (n00 + n01 + n10 + n11),
    }


# ---------------------------------------------------------------------------
# Results loading and aggregation
# ---------------------------------------------------------------------------


def load_results(filepath: str | Path) -> list[dict[str, Any]]:
    """Load evaluation results from a JSONL file."""
    filepath = Path(filepath)
    results: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def aggregate_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate raw results into a summary table.

    Groups by (model, setting, prompt_template) and computes metrics per group.
    """
    groups: dict[tuple, list[dict[str, Any]]] = defaultdict(list)

    for r in results:
        key = (
            r.get("model", "unknown"),
            r.get("setting", "unknown"),
            r.get("prompt_template", "unknown"),
            r.get("context_mode", "present"),
            r.get("run_index", 0),
        )
        groups[key].append(r)

    summary: dict[str, Any] = {}
    for (model, setting, template, context_mode, run_index), group_results in sorted(groups.items()):
        gold = [r["gold_label"] for r in group_results]
        pred = [r["predicted_label"] for r in group_results]

        group_key = f"{model}__{setting}__{template}__{context_mode}__run{run_index}"
        summary[group_key] = {
            "model": model,
            "setting": setting,
            "prompt_template": template,
            "context_mode": context_mode,
            "run_index": run_index,
            "overall": compute_metrics(gold, pred),
            "per_category": compute_per_category_metrics(group_results),
        }

    return summary


def print_results_table(summary: dict[str, Any]) -> None:
    """Print a formatted results comparison table."""
    print(f"\n{'=' * 90}")
    print(f"{'Model':<30} {'Setting':<15} {'Sarc Acc':>10} {'Impl Acc':>10} {'Ind Acc':>10} {'Macro-F1':>10}")
    print(f"{'=' * 90}")

    for key, data in summary.items():
        model = data["model"][:28]
        setting = data["setting"][:13]

        sarc_acc = data["per_category"].get("sarcasm", {}).get("accuracy", 0)
        impl_acc = data["per_category"].get("implicature", {}).get("accuracy", 0)
        ind_acc = data["per_category"].get("indirect_speech_act", {}).get("accuracy", 0)
        macro_f1 = data["overall"]["macro_f1"]

        print(f"{model:<30} {setting:<15} {sarc_acc:>10.3f} {impl_acc:>10.3f} {ind_acc:>10.3f} {macro_f1:>10.3f}")

    print(f"{'=' * 90}\n")
