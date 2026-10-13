"""
Shared deduplication utilities for the pragmatic competence benchmark.

Provides both exact (hash-based) and fuzzy (Jaccard token-overlap) near-
duplicate detection for utterance/context pairs. Used by split.py to prevent
train-test leakage and can be reused by any other preprocessing step.

Usage:
    from src.preprocessing.dedup import find_near_duplicates, normalize_text_for_dedup
"""

import re
from typing import Any


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_text_for_dedup(text: str) -> str:
    """Normalise text for fuzzy duplicate detection.

    Lowercases, strips punctuation and collapses whitespace so that minor
    surface variants (capitalisation, trailing period) are treated as identical.
    """
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _token_set(text: str) -> set[str]:
    """Return the set of normalised word tokens in *text*."""
    return set(normalize_text_for_dedup(text).split())


def jaccard(text_a: str, text_b: str) -> float:
    """Compute the Jaccard token-overlap similarity between two strings.

    Returns a value in [0, 1] where 1 means identical token sets.
    Falls back to 0 when both strings are empty.
    """
    a, b = _token_set(text_a), _token_set(text_b)
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Exact duplicate detection (O(N) hash-based)
# ---------------------------------------------------------------------------


def find_exact_duplicates(
    items: list[dict[str, Any]],
    key_fields: tuple[str, ...] = ("context", "utterance"),
) -> list[tuple[int, int]]:
    """Find items that share an identical normalised key.

    Args:
        items: List of data dicts (must contain the specified *key_fields*).
        key_fields: Tuple of field names whose concatenation forms the key.

    Returns:
        List of (idx_a, idx_b) pairs where idx_b is the duplicate of idx_a.
    """
    seen: dict[str, int] = {}
    duplicates: list[tuple[int, int]] = []

    for idx, item in enumerate(items):
        parts = " ".join(str(item.get(f, "")) for f in key_fields)
        key = normalize_text_for_dedup(parts)
        if key in seen:
            duplicates.append((seen[key], idx))
        else:
            seen[key] = idx

    return duplicates


# ---------------------------------------------------------------------------
# Near-duplicate detection (Jaccard-based, utterance-only)
# ---------------------------------------------------------------------------


def find_near_duplicates(
    items: list[dict[str, Any]],
    threshold: float = 0.85,
    key_fields: tuple[str, ...] = ("context", "utterance"),
) -> list[tuple[int, int]]:
    """Find near-duplicate pairs using Jaccard token overlap.

    This is O(N²) so should only be used on moderately sized lists.
    For large datasets, use find_exact_duplicates for a first pass and
    only apply this to remaining candidates.

    Args:
        items: List of data dicts.
        threshold: Jaccard similarity threshold (0–1). Pairs at or above
            this value are considered near-duplicates.
        key_fields: Fields whose concatenated text is compared.

    Returns:
        List of (idx_a, idx_b) pairs of near-duplicate items.
    """
    # First, find exact duplicates (O(N)) cheaply
    exact = set(find_exact_duplicates(items, key_fields))
    # Jaccard pass for near-duplicates (O(N²)) — only when no exact hits
    texts = [
        " ".join(str(item.get(f, "")) for f in key_fields) for item in items
    ]
    pairs: list[tuple[int, int]] = sorted(exact)
    tokens = [_token_set(text) for text in texts]
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            if (i, j) in exact:
                continue
            a, b = tokens[i], tokens[j]
            if a and b and min(len(a), len(b)) / max(len(a), len(b)) < threshold:
                continue
            score = len(a & b) / len(a | b) if a or b else 1.0
            if score >= threshold:
                pairs.append((i, j))
    return pairs


# ---------------------------------------------------------------------------
# Cross-set leakage detection
# ---------------------------------------------------------------------------


def find_cross_leakage(
    reference: list[dict[str, Any]],
    target: list[dict[str, Any]],
    field: str = "utterance",
) -> list[dict[str, Any]]:
    """Find items in *target* whose utterance is a near-match to *reference*.

    Uses exact normalised-hash matching for speed. Suitable for train/test
    leakage checks.

    Args:
        reference: The "clean" set (e.g., training items).
        target: The set to check for contamination (e.g., test items).
        field: The field to compare (default: ``"utterance"``).

    Returns:
        List of dicts describing each leaked item::

            {
                "target_id": str,
                "reference_id": str,
                "utterance": str,
            }
    """
    reference_fps: dict[str, str] = {}
    for item in reference:
        fp = normalize_text_for_dedup(str(item.get(field, "")))
        reference_fps[fp] = item["id"]

    leaks: list[dict[str, Any]] = []
    for item in target:
        fp = normalize_text_for_dedup(str(item.get(field, "")))
        if fp in reference_fps:
            leaks.append(
                {
                    "target_id": item["id"],
                    "reference_id": reference_fps[fp],
                    field: item.get(field, ""),
                }
            )

    return leaks
