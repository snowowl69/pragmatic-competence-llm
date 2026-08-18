"""
Schema validation tests.

Tests the unified benchmark schema validator (src/preprocessing/schema.py)
with valid items, invalid items, and edge cases.
"""

import json
import tempfile
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing.schema import (
    PRAGMATIC_CATEGORIES,
    VALID_LABELS,
    validate_item,
    validate_item_strict,
    validate_jsonl_file,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_valid_sarcasm_item(**overrides):
    """Create a valid sarcasm benchmark item with optional overrides."""
    item = {
        "id": "sarc_0001",
        "context": "A student completely fails an exam.",
        "speaker": "friend",
        "utterance": "Wow, you absolutely nailed that.",
        "literal_meaning": "The student performed very well.",
        "intended_meaning": "The student performed badly; the speaker is being sarcastic.",
        "pragmatic_category": "sarcasm",
        "label": "sarcastic",
        "answer_options": ["sincere", "sarcastic"],
        "difficulty": "easy",
        "requires_context": True,
        "source": "MUStARD",
        "license": "research-use",
        "split": "test",
        "annotator_notes": "",
    }
    item.update(overrides)
    return item


def make_valid_implicature_item(**overrides):
    """Create a valid implicature benchmark item."""
    item = {
        "id": "impl_0001",
        "context": "A: 'Did you like the movie?'",
        "speaker": "B",
        "utterance": "The popcorn was great.",
        "literal_meaning": "B comments on the popcorn quality.",
        "intended_meaning": "B didn't like the movie — deflecting to the popcorn.",
        "pragmatic_category": "implicature",
        "label": "no",
        "answer_options": ["yes", "no", "uncertain"],
        "difficulty": "medium",
        "requires_context": True,
        "source": "Circa",
        "license": "research-use",
        "split": "train",
        "annotator_notes": "",
    }
    item.update(overrides)
    return item


def make_valid_indirect_item(**overrides):
    """Create a valid indirect speech act benchmark item."""
    item = {
        "id": "ind_0001",
        "context": "Two coworkers in a meeting room.",
        "speaker": "coworker",
        "utterance": "It's really stuffy in here.",
        "literal_meaning": "A statement about the air quality.",
        "intended_meaning": "A request to open a window or turn on the AC.",
        "pragmatic_category": "indirect_speech_act",
        "label": "request",
        "answer_options": ["statement", "request", "question", "suggestion", "complaint"],
        "difficulty": "easy",
        "requires_context": False,
        "source": "hand-curated",
        "license": "CC-BY-4.0",
        "split": "train",
        "annotator_notes": "",
    }
    item.update(overrides)
    return item


# ---------------------------------------------------------------------------
# Valid item tests
# ---------------------------------------------------------------------------


class TestValidItems:
    """Test that well-formed items pass validation."""

    def test_valid_sarcasm_item(self):
        errors = validate_item(make_valid_sarcasm_item())
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_implicature_item(self):
        errors = validate_item(make_valid_implicature_item())
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_indirect_item(self):
        errors = validate_item(make_valid_indirect_item())
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_control_item(self):
        item = make_valid_sarcasm_item(id="ctrl_0001", source="control-item", label="sincere")
        errors = validate_item(item)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_valid_hard_item(self):
        item = make_valid_implicature_item(
            id="hard_0001",
            label="ambiguous",
            answer_options=["yes", "no", "ambiguous"],
            difficulty="hard",
        )
        errors = validate_item(item)
        assert errors == [], f"Expected no errors, got: {errors}"

    def test_all_valid_splits(self):
        for split in ["train", "val", "test"]:
            item = make_valid_sarcasm_item(split=split)
            errors = validate_item(item)
            assert errors == [], f"Split '{split}' should be valid, got: {errors}"

    def test_all_valid_difficulties(self):
        for diff in ["easy", "medium", "hard"]:
            item = make_valid_sarcasm_item(difficulty=diff)
            errors = validate_item(item)
            assert errors == [], f"Difficulty '{diff}' should be valid, got: {errors}"


# ---------------------------------------------------------------------------
# Invalid item tests
# ---------------------------------------------------------------------------


class TestInvalidItems:
    """Test that malformed items are caught."""

    def test_missing_required_field(self):
        item = make_valid_sarcasm_item()
        del item["utterance"]
        errors = validate_item(item)
        assert len(errors) > 0, "Should detect missing required field"

    def test_invalid_category(self):
        item = make_valid_sarcasm_item(pragmatic_category="politeness")
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject invalid category"

    def test_invalid_difficulty(self):
        item = make_valid_sarcasm_item(difficulty="extreme")
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject invalid difficulty"

    def test_invalid_split(self):
        item = make_valid_sarcasm_item(split="holdout")
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject invalid split"

    def test_label_not_in_answer_options(self):
        item = make_valid_sarcasm_item(label="uncertain")
        errors = validate_item(item)
        assert any("cross-field" in e for e in errors), \
            "Should detect label not in answer_options"

    def test_wrong_label_for_category(self):
        item = make_valid_sarcasm_item(label="request")
        errors = validate_item(item)
        assert any("cross-field" in e and "not valid for category" in e for e in errors), \
            "Should detect label invalid for sarcasm category"

    def test_empty_utterance(self):
        item = make_valid_sarcasm_item(utterance="")
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject empty utterance"

    def test_too_few_answer_options(self):
        item = make_valid_sarcasm_item(answer_options=["sarcastic"])
        errors = validate_item(item)
        assert len(errors) > 0, "Should require at least 2 answer options"

    def test_invalid_id_format(self):
        item = make_valid_sarcasm_item(id="sarcasm_1")  # Wrong prefix format
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject ID not matching pattern"

    def test_additional_properties_rejected(self):
        item = make_valid_sarcasm_item()
        item["extra_field"] = "should not be here"
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject additional properties"

    def test_wrong_type_requires_context(self):
        item = make_valid_sarcasm_item(requires_context="yes")  # String instead of bool
        errors = validate_item(item)
        assert len(errors) > 0, "Should reject non-boolean requires_context"


# ---------------------------------------------------------------------------
# Strict validation tests
# ---------------------------------------------------------------------------


class TestStrictValidation:
    """Test the strict (raise-on-error) validator."""

    def test_strict_passes_valid(self):
        validate_item_strict(make_valid_sarcasm_item())  # Should not raise

    def test_strict_raises_on_invalid(self):
        item = make_valid_sarcasm_item()
        del item["utterance"]
        raised = False
        try:
            validate_item_strict(item)
        except Exception:
            raised = True
        assert raised, "Expected validate_item_strict to raise Exception"


# ---------------------------------------------------------------------------
# File validation tests
# ---------------------------------------------------------------------------


class TestFileValidation:
    """Test JSONL file-level validation."""

    def test_valid_file(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        items = [
            make_valid_sarcasm_item(id="sarc_0001"),
            make_valid_implicature_item(id="impl_0001"),
            make_valid_indirect_item(id="ind_0001"),
        ]
        with open(filepath, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        results = validate_jsonl_file(filepath)
        assert results["total"] == 3
        assert results["valid"] == 3
        assert results["invalid"] == 0

    def test_duplicate_ids_detected(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        items = [
            make_valid_sarcasm_item(id="sarc_0001"),
            make_valid_sarcasm_item(id="sarc_0001"),  # Duplicate!
        ]
        with open(filepath, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        results = validate_jsonl_file(filepath)
        assert any("Duplicate" in str(e) for e in results["errors"]), \
            "Should detect duplicate IDs"

    def test_invalid_json_line(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        with open(filepath, "w") as f:
            f.write(json.dumps(make_valid_sarcasm_item()) + "\n")
            f.write("this is not json\n")

        results = validate_jsonl_file(filepath)
        assert results["invalid"] > 0, "Should detect invalid JSON"

    def test_category_counts(self, tmp_path):
        filepath = tmp_path / "test.jsonl"
        items = [
            make_valid_sarcasm_item(id="sarc_0001"),
            make_valid_sarcasm_item(id="sarc_0002"),
            make_valid_implicature_item(id="impl_0001"),
        ]
        with open(filepath, "w") as f:
            for item in items:
                f.write(json.dumps(item) + "\n")

        results = validate_jsonl_file(filepath)
        assert results["category_counts"]["sarcasm"] == 2
        assert results["category_counts"]["implicature"] == 1
