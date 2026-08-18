"""
Main evaluation harness for the pragmatic competence benchmark.

Runs LLM evaluation across the benchmark test set using configurable models,
prompt templates, and settings (zero-shot, few-shot). Stores results in JSONL
format per the Section H schema.

Supports:
  - Hugging Face local models (transformers + peft for LoRA adapters)
  - API models (OpenAI-compatible) via simple HTTP calls
  - Context-present and context-removed modes (for RQ4 ablation)
  - Multiple prompt template variants (for prompt sensitivity control)
  - Deterministic generation (temperature=0, fixed seeds)

Usage:
    # Zero-shot baseline on local model
    python src/evaluation/harness.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --setting zero-shot \\
        --test-file data/benchmark/test.jsonl \\
        --output-dir results/baseline/

    # With LoRA adapter
    python src/evaluation/harness.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --adapter experiments/joint/adapter \\
        --setting zero-shot \\
        --test-file data/benchmark/test.jsonl \\
        --output-dir results/post_ft/

    # Context-removed ablation
    python src/evaluation/harness.py \\
        --model meta-llama/Llama-3.1-8B-Instruct \\
        --setting zero-shot \\
        --no-context \\
        --test-file data/benchmark/test.jsonl \\
        --output-dir results/ablation_no_context/
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Windows console encoding guard
# ---------------------------------------------------------------------------
# On Windows the default console encoding is cp1252, which cannot render many
# Unicode characters used in print statements. Reconfigure stdout/stderr to
# UTF-8 with a safe replacement for any codepoints that still can't be shown.
try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
except Exception:
    pass  # Not critical — best-effort only

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tqdm import tqdm

from src.evaluation.prompts import (
    PROMPT_TEMPLATES,
    format_prompt,
    format_prompt_no_context,
    get_valid_labels,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------


def extract_label(raw_output: str, valid_labels: list[str]) -> str:
    """Extract a predicted label from the model's raw text output.

    Strategy:
      1. Check if the output (stripped, lowered) exactly matches a label.
      2. Check if the first word matches a label.
      3. Find all labels present as full word-boundary matches; prefer the
         longest match to avoid substring collision (e.g. "no" inside "noted").
      4. If nothing matches, return "UNPARSEABLE".

    Note: The previous step-4 bare substring match has been removed because it
    caused false positives, e.g. "no" matching inside "known" or "noted".
    """
    text = raw_output.strip().lower()

    # Step 1: direct exact match
    if text in valid_labels:
        return text

    # Step 2: first word match (handles "Yes, ..." style outputs)
    first_word = text.split()[0] if text.split() else ""
    first_word = re.sub(r"[^a-z]", "", first_word)
    if first_word in valid_labels:
        return first_word

    # Step 3: word-boundary search — prefer longest match to avoid collisions
    found = [
        label for label in valid_labels
        if re.search(rf"\b{re.escape(label)}\b", text)
    ]
    if found:
        return max(found, key=len)  # longest label wins (e.g. "sarcastic" > "is")

    return "UNPARSEABLE"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def score_hf_labels(
    model: Any,
    tokenizer: Any,
    prompt: str,
    valid_labels: list[str],
) -> str:
    """Pick the best label using token log-probabilities (single forward pass).

    Instead of generating text and parsing it with regex, this function feeds
    the prompt to the model and checks which label token (or sequence) has the
    highest probability at the next-token position. This is more reliable than
    generation-based label extraction, especially for short, constrained labels.

    Args:
        model: A loaded HF causal language model (in eval mode).
        tokenizer: Matching tokenizer.
        prompt: The formatted prompt string (without any response text).
        valid_labels: Candidate label strings to compare.

    Returns:
        The label with the highest log-probability score.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    input_ids = inputs["input_ids"].to(model.device)

    best_label = valid_labels[0]
    best_score = float("-inf")

    with torch.no_grad():
        # Get logits for the last token position (= first response token)
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # shape: [vocab_size]
        log_probs = torch.log_softmax(logits, dim=-1)

        for label in valid_labels:
            # Tokenise the label (strip leading space artifact from tokenizer)
            label_ids = tokenizer.encode(
                " " + label,  # leading space matches how generation starts
                add_special_tokens=False,
            )
            if not label_ids:
                continue

            if len(label_ids) == 1:
                # Single-token label: use its log-prob directly
                score = log_probs[label_ids[0]].item()
            else:
                # Multi-token label: sum log-probs of each token conditioned
                # on the previous tokens (teacher-forced).
                score = 0.0
                current_input = input_ids
                for tok_id in label_ids:
                    out = model(current_input)
                    lp = torch.log_softmax(out.logits[0, -1, :], dim=-1)
                    score += lp[tok_id].item()
                    new_tok = torch.tensor([[tok_id]], device=model.device)
                    current_input = torch.cat([current_input, new_tok], dim=1)

            if score > best_score:
                best_score = score
                best_label = label

    return best_label


def load_hf_model(
    model_name: str,
    adapter_path: str | None = None,
    quantize_4bit: bool = True,
) -> tuple[Any, Any]:
    """Load a Hugging Face model (optionally with LoRA adapter and 4-bit quantization).

    Returns (model, tokenizer) tuple.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    print(f"Loading model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for memory efficiency
    if quantize_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    # Load LoRA adapter if provided
    if adapter_path:
        from peft import PeftModel
        print(f"Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    model.eval()
    print(f"Model loaded. Device: {model.device}")
    return model, tokenizer


def generate_hf(
    model: Any,
    tokenizer: Any,
    prompt: str,
    max_new_tokens: int = 20,
) -> str:
    """Generate a response from a Hugging Face model.

    Uses greedy decoding (temperature effectively 0) for determinism.
    """
    import torch

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,  # Greedy decoding = deterministic
            temperature=1.0,  # Ignored when do_sample=False, but set for clarity
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the newly generated tokens
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    return response.strip()


# ---------------------------------------------------------------------------
# API model support (OpenAI-compatible)
# ---------------------------------------------------------------------------


def generate_api(
    prompt: str,
    model_name: str = "gpt-4o-mini",
    api_key: str | None = None,
    max_tokens: int = 20,
    max_retries: int = 3,
) -> str:
    """Call an OpenAI-compatible API for generation.

    Set OPENAI_API_KEY environment variable or pass api_key directly.
    """
    import urllib.request

    api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        return "API_KEY_MISSING"

    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode("utf-8"))
                return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"  API error (attempt {attempt + 1}): {e}. Retrying in {wait}s...")
                time.sleep(wait)
            else:
                return f"API_ERROR: {e}"


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def run_evaluation(
    test_file: str | Path,
    model_name: str,
    setting: str = "zero-shot",
    template_version: str = "v1",
    adapter_path: str | None = None,
    no_context: bool = False,
    use_api: bool = False,
    api_model_name: str = "gpt-4o-mini",
    output_dir: str | Path = "results/",
    quantize_4bit: bool = True,
    run_index: int = 0,
    logit_score: bool = False,
) -> list[dict[str, Any]]:
    """Run the evaluation harness on a test set.

    Args:
        test_file: Path to the test JSONL file
        model_name: HuggingFace model ID or "api" for API models
        setting: "zero-shot" or "few-shot"
        template_version: "v1" or "v2" (prompt template variant)
        adapter_path: Optional path to a LoRA adapter directory
        no_context: If True, strip context (for RQ4 ablation)
        use_api: If True, use API instead of local model
        api_model_name: API model name (only used if use_api=True)
        output_dir: Directory to save results
        quantize_4bit: Whether to load local model in 4-bit
        run_index: Run index for repeated evaluations
        logit_score: If True, use token log-probability scoring instead of
            regex parsing for local models. More accurate but ~2x slower.

    Returns:
        List of result dictionaries
    """
    test_file = Path(test_file)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load test items
    print(f"Loading test items from {test_file}...")
    test_items: list[dict[str, Any]] = []
    with open(test_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                test_items.append(json.loads(line))
    print(f"  Loaded {len(test_items)} items")

    # Load model (local or API)
    model, tokenizer = None, None
    if not use_api:
        model, tokenizer = load_hf_model(model_name, adapter_path, quantize_4bit)

    # Determine template name
    template_prefix = "few_shot" if setting == "few-shot" else "zero_shot"
    template_name = f"{template_prefix}_{template_version}"

    # Model display name for results
    display_name = api_model_name if use_api else model_name.split("/")[-1]
    if adapter_path:
        adapter_name = Path(adapter_path).name
        display_name = f"{display_name}+{adapter_name}"

    # Run evaluation
    results: list[dict[str, Any]] = []
    correct = 0

    print(f"\nRunning evaluation: {display_name} / {setting} / {template_name}")
    print(f"  Context mode: {'removed' if no_context else 'present'}")
    print()

    for item in tqdm(test_items, desc="Evaluating"):
        category = item["pragmatic_category"]
        valid_labels = get_valid_labels(category)

        # Format prompt
        if no_context:
            prompt = format_prompt_no_context(category, template_name, item["utterance"])
        else:
            prompt = format_prompt(category, template_name, item["context"], item["utterance"])

        # Generate prediction
        if use_api:
            raw_output = generate_api(prompt, api_model_name)
            predicted_label = extract_label(raw_output, valid_labels)
        elif logit_score:
            # Logit-based scoring: no generation, single forward pass per item
            raw_output = "[logit-scored]"
            predicted_label = score_hf_labels(model, tokenizer, prompt, valid_labels)
        else:
            raw_output = generate_hf(model, tokenizer, prompt)
            predicted_label = extract_label(raw_output, valid_labels)
        gold_label = item["label"]
        is_correct = predicted_label == gold_label
        if is_correct:
            correct += 1

        # Store result per Section H schema
        result = {
            "model": display_name,
            "setting": setting,
            "prompt_template": template_name,
            "item_id": item["id"],
            "pragmatic_category": category,
            "predicted_label": predicted_label,
            "gold_label": gold_label,
            "correct": is_correct,
            "raw_output": raw_output[:500],  # Truncate for storage
            "run_index": run_index,
            "context_mode": "removed" if no_context else "present",
        }
        results.append(result)

    # Save results
    context_suffix = "_no_context" if no_context else ""
    adapter_suffix = f"_{Path(adapter_path).name}" if adapter_path else ""
    filename = f"{display_name}__{setting}__{template_name}{adapter_suffix}{context_suffix}__run{run_index}.jsonl"
    # Sanitize filename
    filename = re.sub(r"[^\w\-.]", "_", filename)
    output_path = output_dir / filename

    with open(output_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Print summary
    accuracy = correct / len(test_items) if test_items else 0
    print(f"\n{'=' * 60}")
    print(f"RESULTS: {display_name} / {setting} / {template_name}")
    print(f"  Accuracy: {accuracy:.3f} ({correct}/{len(test_items)})")
    print(f"  Saved to: {output_path}")
    print(f"{'=' * 60}\n")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the pragmatic competence evaluation harness."
    )
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--setting", type=str, default="zero-shot",
                        choices=["zero-shot", "few-shot"],
                        help="Evaluation setting")
    parser.add_argument("--template", type=str, default="v1",
                        choices=["v1", "v2"],
                        help="Prompt template version")
    parser.add_argument("--test-file", type=str,
                        default=str(PROJECT_ROOT / "data" / "benchmark" / "test.jsonl"),
                        help="Path to test JSONL file")
    parser.add_argument("--output-dir", type=str,
                        default=str(PROJECT_ROOT / "results" / "baseline"),
                        help="Output directory for results")
    parser.add_argument("--no-context", action="store_true",
                        help="Remove context for ablation study")
    parser.add_argument("--use-api", action="store_true",
                        help="Use OpenAI-compatible API instead of local model")
    parser.add_argument("--api-model", type=str, default="gpt-4o-mini",
                        help="API model name (only with --use-api)")
    parser.add_argument("--no-quantize", action="store_true",
                        help="Disable 4-bit quantization for local models")
    parser.add_argument("--runs", type=int, default=1,
                        help="Number of evaluation runs (for stability measurement)")
    parser.add_argument("--logit-score", action="store_true",
                        help="Use token log-probability scoring instead of regex parsing "
                             "(more accurate, ~2x slower, local models only)")

    args = parser.parse_args()

    for run_idx in range(args.runs):
        run_evaluation(
            test_file=args.test_file,
            model_name=args.model,
            setting=args.setting,
            template_version=args.template,
            adapter_path=args.adapter,
            no_context=args.no_context,
            use_api=args.use_api,
            api_model_name=args.api_model,
            output_dir=args.output_dir,
            quantize_4bit=not args.no_quantize,
            run_index=run_idx,
            logit_score=args.logit_score,
        )


if __name__ == "__main__":
    main()
