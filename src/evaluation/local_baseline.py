"""Real CPU reference experiment. This is a TF-IDF classifier, not an LLM."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from src.evaluation.metrics import compute_metrics, mcnemars_test
from src.evaluation.prompts import get_valid_labels


def text(item, context=True):
    return (item["context"] + "\n" if context else "") + item["utterance"]


def train_and_evaluate(benchmark=ROOT / "data/benchmark/v2", output=ROOT / "results/local"):
    benchmark, output = Path(benchmark), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    splits = {s: [json.loads(line) for line in (benchmark / f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train", "test")}
    bundle, report, records = {}, {"method": "TF-IDF word 1-2 grams + logistic regression; hyperparameters fixed before test", "is_llm": False}, []
    for category in ("sarcasm", "implicature", "indirect_speech_act"):
        train = [x for x in splits["train"] if x["pragmatic_category"] == category]
        test = [x for x in splits["test"] if x["pragmatic_category"] == category]
        gold = [x["label"] for x in test]
        majority = Counter(x["label"] for x in train).most_common(1)[0][0]
        pred_majority = [majority] * len(test)
        models, predictions = {}, {}
        for mode in ("present", "removed"):
            classifier = make_pipeline(TfidfVectorizer(ngram_range=(1, 2), max_features=30000, sublinear_tf=True),
                                       LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced"))
            classifier.fit([text(x, mode == "present") for x in train], [x["label"] for x in train])
            predictions[mode] = classifier.predict([text(x, mode == "present") for x in test]).tolist()
            models[mode] = classifier
            for item, prediction in zip(test, predictions[mode]):
                records.append({"model": "TF-IDF-LogisticRegression", "setting": "supervised-reference", "prompt_template": "none",
                                "context_mode": mode, "run_index": 0, "item_id": item["id"], "pragmatic_category": category,
                                "gold_label": item["label"], "predicted_label": prediction, "correct": prediction == item["label"], "raw_output": prediction})
        labels = get_valid_labels(category)
        report[category] = {"n_train": len(train), "majority": compute_metrics(gold, pred_majority, labels),
                            "with_context": compute_metrics(gold, predictions["present"], labels),
                            "without_context": compute_metrics(gold, predictions["removed"], labels),
                            "context_ablation": mcnemars_test(gold, predictions["removed"], predictions["present"])}
        bundle[category] = {"majority": majority, **models}
    joblib.dump(bundle, output / "classifiers.joblib")
    (output / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output / "predictions.jsonl").write_text("".join(json.dumps(x) + "\n" for x in records), encoding="utf-8")
    print(json.dumps({c: {k: v["accuracy"] for k, v in r.items() if isinstance(v, dict) and "accuracy" in v} for c, r in report.items() if isinstance(r, dict)}, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", default=str(ROOT / "data/benchmark/v2"))
    parser.add_argument("--output", default=str(ROOT / "results/local"))
    args = parser.parse_args()
    train_and_evaluate(args.benchmark, args.output)
