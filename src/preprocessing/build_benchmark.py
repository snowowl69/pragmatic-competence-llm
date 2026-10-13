"""Build benchmark v2 from validated text, with connected leakage groups.

The original committed benchmark is retained for provenance, not training.
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, str(ROOT))
from src.preprocessing.normalize import normalize_mustard, normalize_isarcasm, normalize_circa, normalize_indirect
from src.preprocessing.schema import validate_item
from src.preprocessing.dedup import normalize_text_for_dedup as norm
from src.training.data_formatter import convert_benchmark_to_training_data


def leakage_keys(item):
    keys = [("utterance", norm(item["utterance"]))]
    if item["source"] == "Circa":
        # Different answers to the same question must stay together.
        keys.append(("question", norm(item["context"])))
    elif item["source"] == "MUStARD":
        keys.append(("dialogue", norm(item["context"])))
    return keys


def grouped_split(items, seed=42, train_cap=2000):
    rng = random.Random(seed)
    parents = list(range(len(items)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    seen = {}
    for i, item in enumerate(items):
        group_keys = leakage_keys(item)
        if item["source"] == "Circa":
            group_keys = [key for key in group_keys if key[0] == "question"]
        for key in group_keys:
            if key in seen:
                parents[find(i)] = find(seen[key])
            else:
                seen[key] = i
    groups = defaultdict(list)
    for i, item in enumerate(items):
        groups[find(i)].append(dict(item))
    by_cat = defaultdict(list)
    for group in groups.values():
        cats = {x["pragmatic_category"] for x in group}
        by_cat[tuple(sorted(cats))].append(group)
    splits = {s: [] for s in ("train", "val", "test")}
    for cat_groups in by_cat.values():
        rng.shuffle(cat_groups)
        total = sum(map(len, cat_groups))
        counts = Counter()
        for group in cat_groups:
            split = "test" if counts["test"] < total * .15 else "val" if counts["val"] < total * .10 else "train"
            for item in group:
                item["split"] = split
                splits[split].append(item)
            counts[split] += len(group)
    if train_cap:
        by_category = defaultdict(list)
        for item in splits["train"]:
            by_category[item["pragmatic_category"]].append(item)
        splits["train"] = []
        for values in by_category.values():
            splits["train"].extend(rng.sample(values, min(train_cap, len(values))))
    # Generic replies connect thousands of unrelated Circa questions. Group by
    # question, then exclude reply collisions across splits instead of merging
    # the entire corpus into one giant component.
    owner = {}
    for split in ("train", "val", "test"):
        kept = []
        for item in splits[split]:
            key = norm(item["utterance"])
            if key in owner and owner[key] != split:
                continue
            owner[key] = split
            kept.append(item)
        splits[split] = kept
    for values in splits.values():
        values.sort(key=lambda x: x["id"])
    return splits


def audit_splits(splits):
    owners, ids = {}, set()
    for split, items in splits.items():
        for item in items:
            errors = validate_item(item)
            if errors:
                raise ValueError(f"{item['id']}: {errors}")
            if item["id"] in ids:
                raise ValueError(f"Duplicate ID {item['id']}")
            ids.add(item["id"])
            if item["utterance"].strip().isdigit() and len(item["utterance"].strip()) >= 12:
                raise ValueError(f"Numeric ID used as text: {item['id']}")
            for key in leakage_keys(item):
                if key in owners and owners[key] != split:
                    raise ValueError(f"Cross-split leakage: {item['id']}")
                owners[key] = split
    return {split: {"size": len(items), "categories": dict(Counter(x["pragmatic_category"] for x in items)),
                    "labels": dict(Counter(x["pragmatic_category"] + ":" + x["label"] for x in items)),
                    "sources": dict(Counter(x["source"] for x in items))}
            for split, items in splits.items()}


def build(output_dir=ROOT / "data/benchmark/v2", seed=42):
    items = normalize_mustard() + normalize_isarcasm() + normalize_circa() + normalize_indirect()
    unique, seen = [], set()
    for item in items:
        key = (norm(item["context"]), norm(item["utterance"]), item["pragmatic_category"], item["label"])
        if key not in seen:
            unique.append(item)
            seen.add(key)
    splits = grouped_split(unique, seed)
    report = audit_splits(splits)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for split, values in splits.items():
        path = output_dir / f"{split}.jsonl"
        path.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")
        checksums[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        if split != "test":
            for condition, categories in {"joint": None, "sarcasm": ["sarcasm"], "implicature": ["implicature"], "indirect": ["indirect_speech_act"]}.items():
                convert_benchmark_to_training_data(path, ROOT / f"data/training/v2/{split}_instructions_{condition}.jsonl", categories)
    manifest = {"version": 2, "seed": seed, "splits": report, "sha256": checksums,
                "excluded": "Committed iSarcasm files contain IDs without tweet text; excluded.",
                "policy": "Circa questions and MUStARD dialogue contexts are grouped. Exact reply collisions across splits are excluded. Not source-dataset-disjoint.",
                "limitations": ["No new independent human annotation", "Public pretraining contamination cannot be ruled out", "Conditional yes is mapped to uncertain", "Near duplicates require manual review"]}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(ROOT / "data/benchmark/v2"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    build(args.output_dir, args.seed)
