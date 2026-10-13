import json
from pathlib import Path
import pytest
from src.preprocessing.build_benchmark import grouped_split, audit_splits
from src.preprocessing.dedup import find_near_duplicates
from src.preprocessing import normalize
from src.evaluation.harness import extract_label
from src.evaluation.metrics import aggregate_results, mcnemars_test
from src.analysis.significance import _load_all_results

ROOT = Path(__file__).resolve().parents[1]


def test_circa_literal_quotes_preserved(tmp_path, monkeypatch):
    directory = tmp_path / "circa"
    directory.mkdir()
    (directory / "circa-data.tsv").write_text('id\tcontext\tquestion-X\tcanquestion-X\tanswer-Y\tjudgements\tgoldstandard1\tgoldstandard2\n0\twork\t"Do you work?"\twork\t"I do."\tYes\tYes\tYes\n', encoding="utf-8")
    monkeypatch.setattr(normalize, "RAW_DATA_DIR", tmp_path)
    item = normalize.normalize_circa()[0]
    assert item["utterance"] == '"I do."'
    assert item["label"] == "yes"


def test_tweet_ids_are_excluded(tmp_path, monkeypatch):
    directory = tmp_path / "isarcasm"
    directory.mkdir()
    (directory / "ids.csv").write_text("tweet_id,sarcasm_label\n933721764970057729,sarcastic\n")
    monkeypatch.setattr(normalize, "RAW_DATA_DIR", tmp_path)
    assert normalize.normalize_isarcasm() == []


def test_valid_tweet_labels_are_preserved(tmp_path, monkeypatch):
    directory = tmp_path / "isarcasm"
    directory.mkdir()
    (directory / "text.csv").write_text("text,sarcasm_label\nWhat a perfect disaster,sarcastic\nHello,not_sarcastic\n")
    monkeypatch.setattr(normalize, "RAW_DATA_DIR", tmp_path)
    assert [x["label"] for x in normalize.normalize_isarcasm()] == ["sarcastic", "sincere"]


def test_fuzzy_duplicates_not_skipped_when_exact_exists():
    items = [{"context": "", "utterance": x} for x in ("one two three four five", "one two three four five", "one two three four five six")]
    assert (0, 2) in find_near_duplicates(items, .8)


def test_explicit_label_overrides_reasoning():
    assert extract_label("Label: sincere\nThis is not sarcastic.", ["sincere", "sarcastic"]) == "sincere"


def test_context_modes_and_repeats_not_pooled():
    rows = [{"model": "test", "setting": "zero-shot", "prompt_template": "v1", "pragmatic_category": "sarcasm", "gold_label": "sincere", "predicted_label": "sincere", "context_mode": mode, "run_index": index} for mode in ("present", "removed") for index in (0, 1)]
    assert len(aggregate_results(rows)) == 4


def test_repeated_paired_results_rejected(tmp_path):
    record = {"item_id": "x"}
    (tmp_path / "a.jsonl").write_text(json.dumps(record) + "\n" + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="repeated"):
        _load_all_results(tmp_path)


def test_mcnemar_nonempty_and_json_serializable():
    with pytest.raises(ValueError):
        mcnemars_test([], [], [])
    json.dumps(mcnemars_test(["yes"] * 30, ["yes"] * 30, ["no"] * 30))


def test_frozen_benchmark_has_no_leakage():
    import hashlib
    splits = {s: [json.loads(x) for x in (ROOT / f"data/benchmark/v2/{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train", "val", "test")}
    report = audit_splits(splits)
    assert all(len(report[s]["categories"]) == 3 for s in report)
    manifest = json.loads((ROOT / "data/benchmark/v2/manifest.json").read_text())
    for filename, digest in manifest["sha256"].items():
        assert hashlib.sha256((ROOT / "data/benchmark/v2" / filename).read_bytes()).hexdigest() == digest


def test_preset_same_utterance_different_context():
    from app.demo import PRESET_EXAMPLES, match_preset, predict
    sincere = PRESET_EXAMPLES[1]
    assert match_preset(sincere["context"], sincere["utterance"], sincere["category"])["gold_label"] == "sincere"
    assert "**Gold Label:** sincere" in predict(sincere["context"], sincere["utterance"], sincere["category"])


def test_evaluation_resumes_after_transport_failure(tmp_path, monkeypatch):
    from src.evaluation import harness
    items = [{"id": f"sarc_{i:04d}", "pragmatic_category": "sarcasm", "context": "A greeting", "utterance": "Hello", "label": "sincere"} for i in range(3)]
    path = tmp_path / "test.jsonl"
    path.write_text("".join(json.dumps(x) + "\n" for x in items))
    calls = []
    def flaky(*args):
        calls.append(args)
        if len(calls) == 2:
            raise RuntimeError("connection failed")
        return "sincere"
    monkeypatch.setattr(harness, "generate_api", flaky)
    with pytest.raises(RuntimeError):
        harness.run_evaluation(path, "api", use_api=True, output_dir=tmp_path / "results")
    resumed = []
    monkeypatch.setattr(harness, "generate_api", lambda *args: resumed.append(args) or "sincere")
    results = harness.run_evaluation(path, "api", use_api=True, output_dir=tmp_path / "results")
    assert len(results) == 3
    assert len(resumed) == 2
