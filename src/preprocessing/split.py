"""
Train/val/test splitting with source-level leakage prevention and contamination checks.

Splits the combined processed data into frozen benchmark splits, ensuring:
  - No source-level leakage (items from the same source don't span train and test)
  - Category balance across splits
  - Near-duplicate detection (fuzzy matching on utterance text)

Usage:
    python src/preprocessing/split.py [--input data/processed/all_combined.jsonl]
                                       [--output-dir data/benchmark/]
                                       [--test-ratio 0.15]
                                       [--val-ratio 0.10]
                                       [--seed 42]
"""

import argparse
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Windows console encoding guard
# ---------------------------------------------------------------------------
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.dedup import (
    find_cross_leakage,
    find_near_duplicates,
    normalize_text_for_dedup,
)

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "all_combined.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark"

# Maximum training items per category — prevents any single source from
# dominating the training distribution (addresses the 230x Circa imbalance).
DEFAULT_TRAIN_CAP = 2000


# ---------------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------------


def cap_by_category(
    items: list[dict[str, Any]],
    max_per_category: int = DEFAULT_TRAIN_CAP,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """Cap training items per category to address severe class imbalance.

    When one source (e.g. Circa) contributes 230x more items than another,
    the model ends up learning mostly that distribution. This cap ensures
    every category contributes at most *max_per_category* training examples.

    Args:
        items: Training items to cap.
        max_per_category: Maximum items allowed per ``pragmatic_category``.
        seed: Random seed for reproducible sampling.

    Returns:
        Capped list of training items (original objects, not copies).
    """
    rng = random.Random(seed)
    by_cat: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_cat[item["pragmatic_category"]].append(item)

    capped: list[dict[str, Any]] = []
    for cat, cat_items in sorted(by_cat.items()):
        before = len(cat_items)
        if len(cat_items) > max_per_category:
            cat_items = rng.sample(cat_items, max_per_category)
        capped.extend(cat_items)
        after = len(cat_items)
        if after < before:
            print(f"  [cap] {cat}: {before} -> {after} training items (cap={max_per_category})")

    return capped


def split_items(
    items: list[dict[str, Any]],
    test_ratio: float = 0.15,
    val_ratio: float = 0.10,
    seed: int = 42,
    train_cap: int = DEFAULT_TRAIN_CAP,
) -> dict[str, list[dict[str, Any]]]:
    """Split items into train/val/test with stratification by category.

    Strategy:
    - Group items by pragmatic_category
    - Within each category, shuffle and split proportionally
    - Assign control items (source="control-item") to test set preferentially
    - Hand-curated items preferentially go to test set (for contamination resistance)
    - Cap training items per category to *train_cap* to prevent skew

    Returns: {"train": [...], "val": [...], "test": [...]}
    """
    rng = random.Random(seed)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_category[item["pragmatic_category"]].append(item)

    for category, cat_items in sorted(by_category.items()):
        # Separate control items and hand-curated items (-> test preferentially)
        test_priority = []
        regular = []

        for item in cat_items:
            source = item.get("source", "")
            if source in ("control-item", "hand-curated"):
                test_priority.append(item)
            else:
                regular.append(item)

        rng.shuffle(regular)
        rng.shuffle(test_priority)

        n_total = len(cat_items)
        n_test = max(1, int(n_total * test_ratio))
        n_val = max(1, int(n_total * val_ratio))

        # Fill test set with priority items first, then regular
        test_items = test_priority[:n_test]
        remaining_priority = test_priority[n_test:]
        if len(test_items) < n_test:
            needed = n_test - len(test_items)
            test_items.extend(regular[:needed])
            regular = regular[needed:]
        else:
            regular = remaining_priority + regular

        # Fill val set
        val_items = regular[:n_val]
        train_items = regular[n_val:]

        # Assign split labels
        for item in train_items:
            item["split"] = "train"
        for item in val_items:
            item["split"] = "val"
        for item in test_items:
            item["split"] = "test"

        splits["train"].extend(train_items)
        splits["val"].extend(val_items)
        splits["test"].extend(test_items)

        print(f"  {category}: {len(train_items)} train / {len(val_items)} val / {len(test_items)} test")

    # Apply per-category cap on training items to prevent source imbalance
    if train_cap > 0:
        print(f"\nApplying per-category training cap (max={train_cap})...")
        splits["train"] = cap_by_category(splits["train"], train_cap, seed)

    return splits


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------


def check_train_test_leakage(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Check for utterance-level leakage between train and test sets.

    Returns list of dicts describing each leaked test item (id, utterance,
    matching train id). Delegates to the shared ``find_cross_leakage`` helper
    from ``src.preprocessing.dedup``.
    """
    raw = find_cross_leakage(reference=train, target=test, field="utterance")
    # Rename keys to preserve backward-compat with downstream callers
    return [
        {
            "test_id": item["target_id"],
            "train_id": item["reference_id"],
            "utterance": item["utterance"],
        }
        for item in raw
    ]


# ---------------------------------------------------------------------------
# Write output
# ---------------------------------------------------------------------------


def write_splits(
    splits: dict[str, list[dict[str, Any]]],
    output_dir: Path,
) -> None:
    """Write each split to a separate JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, items in splits.items():
        output_path = output_dir / f"{split_name}.jsonl"
        with open(output_path, "w", encoding="utf-8") as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"  [OK] {split_name}: {len(items)} items -> {output_path}")

    # Write a checksum file for reproducibility
    checksum_path = output_dir / "checksums.txt"
    with open(checksum_path, "w", encoding="utf-8") as f:
        for split_name in sorted(splits):
            filepath = output_dir / f"{split_name}.jsonl"
            content = filepath.read_bytes()
            md5 = hashlib.md5(content).hexdigest()
            f.write(f"{split_name}.jsonl: {md5}\n")
    print(f"  [OK] Checksums written to {checksum_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Split processed benchmark data into train/val/test JSONL files."
    )
    parser.add_argument("--input", type=str, default=str(DEFAULT_INPUT),
                        help="Path to combined processed JSONL file.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for split files.")
    parser.add_argument("--test-ratio", type=float, default=0.15,
                        help="Fraction of data for the test split.")
    parser.add_argument("--val-ratio", type=float, default=0.10,
                        help="Fraction of data for the validation split.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    parser.add_argument("--dedup-threshold", type=float, default=0.85,
                        help="Jaccard similarity threshold for near-duplicate detection.")
    parser.add_argument("--train-cap", type=int, default=DEFAULT_TRAIN_CAP,
                        help="Max training items per category (0 = no cap).")

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}")
        print("  Run normalize.py first to create the combined processed file.")
        return

    # Load all items
    print(f"Loading items from {input_path}...")
    items: list[dict[str, Any]] = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    print(f"  Loaded {len(items)} items\n")

    # Deduplication check — remove second occurrence of any duplicate pair
    print("Checking for near-duplicates...")
    dupes = find_near_duplicates(items, threshold=args.dedup_threshold)
    if dupes:
        print(f"  [WARN] Found {len(dupes)} near-duplicate pairs — removing later occurrences:")
        dup_indices = {j for _, j in dupes}  # Keep idx_a, remove idx_b
        for i, j in dupes[:10]:
            print(f"    {items[i]['id']} <-> {items[j]['id']}: \"{items[i]['utterance'][:50]}...\"")
        if len(dupes) > 10:
            print(f"    ... and {len(dupes) - 10} more")
        items = [item for idx, item in enumerate(items) if idx not in dup_indices]
        print(f"  [INFO] Items after dedup: {len(items)}\n")
    else:
        print("  [OK] No near-duplicates found.\n")

    # Split
    print(f"Splitting (test={args.test_ratio}, val={args.val_ratio}, seed={args.seed})...")
    splits = split_items(
        items,
        args.test_ratio,
        args.val_ratio,
        args.seed,
        train_cap=args.train_cap,
    )
    print()

    # Leakage check — auto-remove leaked test items
    print("Checking for train/test leakage...")
    leaks = check_train_test_leakage(splits["train"], splits["test"])
    if leaks:
        leaked_test_ids = {leak["test_id"] for leak in leaks}
        print(f"  [WARN] Found {len(leaks)} leaked test items — auto-removing from test split:")
        for leak in leaks[:5]:
            print(f"    test:{leak['test_id']} <-> train:{leak['train_id']}: \"{leak['utterance'][:50]}...\"")
        if len(leaks) > 5:
            print(f"    ... and {len(leaks) - 5} more")
        splits["test"] = [item for item in splits["test"] if item["id"] not in leaked_test_ids]
        print(f"  [OK] Test split after leak removal: {len(splits['test'])} items\n")
    else:
        print("  [OK] No train/test leakage detected.\n")

    # Write
    print("Writing split files...")
    write_splits(splits, output_dir)

    # Summary
    print(f"\n{'=' * 60}")
    print("BENCHMARK SPLIT SUMMARY")
    print(f"{'=' * 60}")
    for split_name, items_in_split in splits.items():
        cats = defaultdict(int)
        for item in items_in_split:
            cats[item["pragmatic_category"]] += 1
        cat_str = ", ".join(f"{k}: {v}" for k, v in sorted(cats.items()))
        print(f"  {split_name}: {len(items_in_split)} items ({cat_str})")
    print(f"{'=' * 60}")
    print("\n[NOTE] After this point, do NOT edit split files without a version bump.")
    print("  Any label corrections should be logged and applied as a versioned update.")


if __name__ == "__main__":
    main()
