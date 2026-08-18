"""
Visualization module — transfer matrix heatmaps, failure charts, confusion matrices.

Generates publication-quality plots for the research paper and demo.
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Color scheme
# ---------------------------------------------------------------------------

COLORS = {
    "sarcasm": "#E74C3C",
    "implicature": "#3498DB",
    "indirect_speech_act": "#2ECC71",
    "background": "#F8F9FA",
    "grid": "#E0E0E0",
}


# ---------------------------------------------------------------------------
# Transfer matrix heatmap
# ---------------------------------------------------------------------------


def plot_transfer_matrix(
    matrix_data: dict[str, Any],
    output_path: str | Path = "results/plots/transfer_matrix.png",
    title: str = "Cross-Category Transfer Matrix",
) -> None:
    """Plot the transfer matrix as an annotated heatmap.

    This is the single most important visualization for the research presentation.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = matrix_data["categories"]
    conditions = matrix_data.get("conditions", [])

    # Include baseline as a row
    all_rows = ["baseline"] + conditions if matrix_data.get("baseline") else conditions
    cat_short = {
        "sarcasm": "Sarcasm",
        "implicature": "Implicature",
        "indirect_speech_act": "Indirect SA",
    }

    # Build the numpy matrix
    data = np.zeros((len(all_rows), len(categories)))
    for i, row_name in enumerate(all_rows):
        if row_name == "baseline":
            source = matrix_data["baseline"]
        else:
            source = matrix_data["matrix"].get(row_name, {})

        for j, cat in enumerate(categories):
            data[i, j] = source.get(cat, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(8, max(4, len(all_rows) * 0.8 + 2)))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)

    # Labels
    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels([cat_short.get(c, c) for c in categories], fontsize=11)
    ax.set_yticks(range(len(all_rows)))
    row_labels = ["Baseline (0-shot)" if r == "baseline" else f"Trained: {r}" for r in all_rows]
    ax.set_yticklabels(row_labels, fontsize=11)

    ax.set_xlabel("Test Category", fontsize=12, fontweight="bold")
    ax.set_ylabel("Training Condition", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Annotate cells with values
    for i in range(len(all_rows)):
        for j in range(len(categories)):
            val = data[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Accuracy", fontsize=11)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Transfer matrix heatmap saved to {output_path}")


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------


def plot_confusion_matrix(
    cm: list[list[int]],
    label_names: list[str],
    title: str = "Confusion Matrix",
    output_path: str | Path = "results/plots/confusion_matrix.png",
) -> None:
    """Plot a confusion matrix with annotations."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cm_array = np.array(cm)
    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(cm_array, cmap="Blues", interpolation="nearest")

    ax.set_xticks(range(len(label_names)))
    ax.set_xticklabels(label_names, rotation=45, ha="right")
    ax.set_yticks(range(len(label_names)))
    ax.set_yticklabels(label_names)

    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")

    # Annotate
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            val = cm_array[i, j]
            text_color = "white" if val > cm_array.max() / 2 else "black"
            ax.text(j, i, str(val), ha="center", va="center",
                    fontsize=12, color=text_color)

    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Confusion matrix saved to {output_path}")


# ---------------------------------------------------------------------------
# Failure taxonomy stacked bar chart
# ---------------------------------------------------------------------------


def plot_failure_taxonomy(
    aggregated: dict[str, dict[str, int]],
    output_path: str | Path = "results/plots/failure_taxonomy.png",
    title: str = "Failure Type Distribution by Category",
) -> None:
    """Plot failure taxonomy as a stacked bar chart."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = sorted(aggregated.keys())
    all_types = sorted(set().union(*[set(d.keys()) for d in aggregated.values()]))

    # Build data matrix
    data = np.zeros((len(all_types), len(categories)))
    for j, cat in enumerate(categories):
        for i, ftype in enumerate(all_types):
            data[i, j] = aggregated[cat].get(ftype, 0)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(categories))
    width = 0.6
    bottom = np.zeros(len(categories))

    colors = plt.cm.Set3(np.linspace(0, 1, len(all_types)))

    for i, ftype in enumerate(all_types):
        ax.bar(x, data[i], width, bottom=bottom, label=ftype, color=colors[i])
        bottom += data[i]

    cat_short = {
        "sarcasm": "Sarcasm",
        "implicature": "Implicature",
        "indirect_speech_act": "Indirect SA",
    }

    ax.set_xticks(x)
    ax.set_xticklabels([cat_short.get(c, c) for c in categories], fontsize=11)
    ax.set_ylabel("Error Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Failure taxonomy chart saved to {output_path}")


# ---------------------------------------------------------------------------
# Baseline vs. fine-tuned comparison bar chart
# ---------------------------------------------------------------------------


def plot_baseline_vs_finetuned(
    baseline_accs: dict[str, float],
    finetuned_accs: dict[str, float],
    output_path: str | Path = "results/plots/baseline_vs_finetuned.png",
    title: str = "Baseline vs. Fine-Tuned Accuracy",
) -> None:
    """Plot a grouped bar chart comparing baseline and fine-tuned accuracy per category."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    categories = sorted(set(baseline_accs.keys()) | set(finetuned_accs.keys()))
    cat_short = {
        "sarcasm": "Sarcasm",
        "implicature": "Implicature",
        "indirect_speech_act": "Indirect SA",
    }

    x = np.arange(len(categories))
    width = 0.35

    baseline_vals = [baseline_accs.get(c, 0) for c in categories]
    finetuned_vals = [finetuned_accs.get(c, 0) for c in categories]

    fig, ax = plt.subplots(figsize=(8, 5))

    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline (0-shot)",
                   color="#95A5A6", edgecolor="white")
    bars2 = ax.bar(x + width / 2, finetuned_vals, width, label="Fine-tuned (QLoRA)",
                   color="#2ECC71", edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels([cat_short.get(c, c) for c in categories], fontsize=11)
    ax.set_ylabel("Accuracy", fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                    f"{height:.2f}", ha="center", va="bottom", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart saved to {output_path}")
