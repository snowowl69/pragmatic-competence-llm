"""
Evaluation harness sanity tests.

Tests prompt formatting, label extraction, and metrics computation
to ensure the evaluation pipeline produces correct results.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.prompts import (
    PROMPT_TEMPLATES,
    format_prompt,
    format_prompt_no_context,
    get_valid_labels,
)
from src.evaluation.harness import extract_label
from src.evaluation.metrics import compute_metrics, mcnemars_test


# ---------------------------------------------------------------------------
# Prompt formatting tests
# ---------------------------------------------------------------------------


class TestPromptFormatting:
    """Test prompt template formatting."""

    def test_sarcasm_zero_shot_v1(self):
        prompt = format_prompt("sarcasm", "zero_shot_v1", "It's raining.", "Great weather!")
        assert "It's raining." in prompt
        assert "Great weather!" in prompt
        assert "sincere" in prompt.lower() or "sarcastic" in prompt.lower()

    def test_implicature_zero_shot_v1(self):
        prompt = format_prompt("implicature", "zero_shot_v1", "A: 'Did you like it?'", "The popcorn was great.")
        assert "popcorn" in prompt
        assert "yes" in prompt.lower() or "no" in prompt.lower()

    def test_indirect_zero_shot_v1(self):
        prompt = format_prompt("indirect_speech_act", "zero_shot_v1", "Office meeting.", "It's cold in here.")
        assert "cold" in prompt
        assert "request" in prompt.lower() or "statement" in prompt.lower()

    def test_all_templates_format_without_error(self):
        """Every registered template should format without errors."""
        for category, templates in PROMPT_TEMPLATES.items():
            for template_name in templates:
                prompt = format_prompt(category, template_name, "Test context", "Test utterance")
                assert len(prompt) > 50, f"Prompt too short for {category}/{template_name}"

    def test_no_context_mode(self):
        prompt = format_prompt_no_context("sarcasm", "zero_shot_v1", "Great weather!")
        assert "No context provided" in prompt
        assert "Great weather!" in prompt

    def test_invalid_category_raises(self):
        raised = False
        try:
            format_prompt("nonexistent", "zero_shot_v1", "ctx", "utt")
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for invalid category"

    def test_invalid_template_raises(self):
        raised = False
        try:
            format_prompt("sarcasm", "nonexistent_v99", "ctx", "utt")
        except ValueError:
            raised = True
        assert raised, "Expected ValueError for invalid template"

    def test_two_prompt_versions_differ(self):
        """V1 and V2 should be different (prompt sensitivity control)."""
        v1 = format_prompt("sarcasm", "zero_shot_v1", "context", "utterance")
        v2 = format_prompt("sarcasm", "zero_shot_v2", "context", "utterance")
        assert v1 != v2, "V1 and V2 should be different prompt phrasings"


# ---------------------------------------------------------------------------
# Label extraction tests
# ---------------------------------------------------------------------------


class TestLabelExtraction:
    """Test the label extraction from raw model outputs."""

    def test_exact_match(self):
        assert extract_label("sarcastic", ["sincere", "sarcastic"]) == "sarcastic"
        assert extract_label("sincere", ["sincere", "sarcastic"]) == "sincere"

    def test_case_insensitive(self):
        assert extract_label("SARCASTIC", ["sincere", "sarcastic"]) == "sarcastic"
        assert extract_label("Sincere", ["sincere", "sarcastic"]) == "sincere"

    def test_with_surrounding_text(self):
        assert extract_label("The answer is sarcastic.", ["sincere", "sarcastic"]) == "sarcastic"
        assert extract_label("I believe this is sincere.", ["sincere", "sarcastic"]) == "sincere"

    def test_first_word_extraction(self):
        assert extract_label("sarcastic\n\nBecause the context...", ["sincere", "sarcastic"]) == "sarcastic"

    def test_unparseable(self):
        assert extract_label("I'm not sure what to say here.", ["sincere", "sarcastic"]) == "UNPARSEABLE"
        assert extract_label("", ["sincere", "sarcastic"]) == "UNPARSEABLE"

    def test_multiword_labels(self):
        assert extract_label("request", ["statement", "request", "question"]) == "request"
        assert extract_label("It's a request.", ["statement", "request", "question"]) == "request"

    def test_implicature_labels(self):
        assert extract_label("yes", ["yes", "no", "uncertain"]) == "yes"
        assert extract_label("no", ["yes", "no", "uncertain"]) == "no"
        assert extract_label("uncertain", ["yes", "no", "uncertain"]) == "uncertain"


# ---------------------------------------------------------------------------
# Metrics tests
# ---------------------------------------------------------------------------


class TestMetrics:
    """Test metrics computation."""

    def test_perfect_accuracy(self):
        gold = ["a", "b", "c", "a"]
        pred = ["a", "b", "c", "a"]
        result = compute_metrics(gold, pred)
        assert result["accuracy"] == 1.0
        assert result["macro_f1"] == 1.0

    def test_zero_accuracy(self):
        gold = ["a", "b", "c"]
        pred = ["b", "c", "a"]
        result = compute_metrics(gold, pred)
        assert result["accuracy"] == 0.0

    def test_partial_accuracy(self):
        gold = ["a", "b", "a", "b"]
        pred = ["a", "a", "a", "b"]
        result = compute_metrics(gold, pred)
        assert result["accuracy"] == 0.75


# ---------------------------------------------------------------------------
# McNemar's test tests
# ---------------------------------------------------------------------------


class TestMcNemarsTest:
    """Test McNemar's significance test."""

    def test_identical_predictions(self):
        gold = ["a", "b", "a", "b"]
        pred_a = ["a", "a", "a", "b"]
        pred_b = ["a", "a", "a", "b"]
        result = mcnemars_test(gold, pred_a, pred_b)
        assert result["p_value"] == 1.0, "Identical predictions should have p=1.0"

    def test_different_predictions(self):
        gold = ["a"] * 50 + ["b"] * 50
        pred_a = ["a"] * 50 + ["a"] * 50  # Always predicts "a"
        pred_b = ["a"] * 50 + ["b"] * 50  # Perfect
        result = mcnemars_test(gold, pred_a, pred_b)
        assert result["p_value"] < 0.05, "Very different predictions should be significant"
        assert result["model_b_accuracy"] > result["model_a_accuracy"]

    def test_returns_required_fields(self):
        gold = ["a", "b"]
        pred_a = ["a", "a"]
        pred_b = ["b", "b"]
        result = mcnemars_test(gold, pred_a, pred_b)
        assert "p_value" in result
        assert "significant_at_005" in result
        assert "contingency_table" in result
        assert "method" in result


# ---------------------------------------------------------------------------
# Valid labels tests
# ---------------------------------------------------------------------------


class TestValidLabels:
    """Test the valid labels lookup."""

    def test_sarcasm_labels(self):
        labels = get_valid_labels("sarcasm")
        assert "sincere" in labels
        assert "sarcastic" in labels

    def test_implicature_labels(self):
        labels = get_valid_labels("implicature")
        assert "yes" in labels
        assert "no" in labels
        assert "uncertain" in labels

    def test_indirect_labels(self):
        labels = get_valid_labels("indirect_speech_act")
        assert "request" in labels
        assert "statement" in labels
