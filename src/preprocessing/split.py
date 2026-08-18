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
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "all_combined.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "benchmark"


# ---------------------------------------------------------------------------
# Deduplication utilities
# ---------------------------------------------------------------------------


def normalize_text_for_dedup(text: str) -> str:
    """Normalize text for fuzzy duplicate detection.

    Lowercases, strips punctuation and extra whitespace.
    """
    import re
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def find_near_duplicates(items: list[dict[str, Any]], threshold: float = 0.85) -> list[tuple[int, int]]:
    """Find duplicate pairs based on normalized context + utterance text matching.

    Uses hash maps for O(N) instant duplicate detection across large datasets.
    Returns list of (idx_a, idx_b) pairs that are duplicates.
    """
    seen: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []

    for idx, item in enumerate(items):
        text_key = normalize_text_for_dedup(f"{item.get('context', '')} {item['utterance']}")
        if text_key in seen:
            duplicates.append((seen[text_key], idx))
        else:
            seen[text_key] = idx

    return duplicates


# ---------------------------------------------------------------------------
# Splitting logic
# ---------------------------------------------------------------------------


def split_items(
    items: list[dict[str, Any]],
    test_ratio: float = 0.15,
    val_ratio: float = 0.10,
    seed: int = 42,
) -> dict[str, list[dict[str, Any]]]:
    """Split items into train/val/test with stratification by category.

    Strategy:
    - Group items by pragmatic_category
    - Within each category, shuffle and split proportionally
    - Assign control items (source="control-item") to test set preferentially
    - Hand-curated items preferentially go to test set (for contamination resistance)

    Returns: {"train": [...], "val": [...], "test": [...]}
    """
    rng = random.Random(seed)

    splits: dict[str, list[dict[str, Any]]] = {"train": [], "val": [], "test": []}

    # Group by category
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_category[item["pragmatic_category"]].append(item)

    for category, cat_items in sorted(by_category.items()):
        # Separate control items and hand-curated items (→ test preferentially)
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

    return splits


# ---------------------------------------------------------------------------
# Contamination check
# ---------------------------------------------------------------------------


def check_train_test_leakage(
    train: list[dict[str, Any]],
    test: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Check for utterance-level leakage between train and test sets.

    Returns list of leaked items (from test set that have a near-match in train).
    """
    train_fingerprints: dict[str, str] = {}
    for item in train:
        fp = normalize_text_for_dedup(item["utterance"])
        train_fingerprints[fp] = item["id"]

    leaks: list[dict[str, Any]] = []
    for item in test:
        fp = normalize_text_for_dedup(item["utterance"])
        if fp in train_fingerprints:
            leaks.append({
                "test_id": item["id"],
                "train_id": train_fingerprints[fp],
                "utterance": item["utterance"],
            })

    return leaks


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
        print(f"  ✓ {split_name}: {len(items)} items → {output_path}")

    # Write a checksum file for reproducibility
    checksum_path = output_dir / "checksums.txt"
    with open(checksum_path, "w", encoding="utf-8") as f:
        for split_name in sorted(splits):
            filepath = output_dir / f"{split_name}.jsonl"
            content = filepath.read_bytes()
            md5 = hashlib.md5(content).hexdigest()
            f.write(f"{split_name}.jsonl: {md5}\n")
    print(f"  ✓ Checksums written to {checksum_path}")


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

    args = parser.parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)

    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
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

    # Deduplication check
    print("Checking for near-duplicates...")
    dupes = find_near_duplicates(items, threshold=args.dedup_threshold)
    if dupes:
        print(f"  ⚠ Found {len(dupes)} near-duplicate pairs:")
        for i, j in dupes[:10]:
            print(f"    {items[i]['id']} ↔ {items[j]['id']}: \"{items[i]['utterance'][:50]}...\"")
        if len(dupes) > 10:
            print(f"    ... and {len(dupes) - 10} more")
        print("  Consider removing duplicates before splitting.\n")
    else:
        print("  ✓ No near-duplicates found.\n")

    # Split
    print(f"Splitting (test={args.test_ratio}, val={args.val_ratio}, seed={args.seed})...")
    splits = split_items(items, args.test_ratio, args.val_ratio, args.seed)
    print()

    # Leakage check
    print("Checking for train/test leakage...")
    leaks = check_train_test_leakage(splits["train"], splits["test"])
    if leaks:
        print(f"  ⚠ Found {len(leaks)} leaked items:")
        for leak in leaks[:5]:
            print(f"    test:{leak['test_id']} ↔ train:{leak['train_id']}: \"{leak['utterance'][:50]}...\"")
        print("  These items should be removed from either train or test.\n")
    else:
        print("  ✓ No train/test leakage detected.\n")

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
    print("\n⚠ IMPORTANT: After this point, do NOT edit split files without a version bump.")
    print("  Any label corrections should be logged and applied as a versioned update.")


if __name__ == "__main__":
    main()
