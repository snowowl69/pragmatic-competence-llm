"""
Gradio demo application — Baseline vs. Fine-Tuned side-by-side comparison.

Shows the difference between a base LLM's pragmatic interpretation and the
QLoRA fine-tuned model's interpretation, for sarcasm, implicature, and
indirect speech acts.

Usage:
    python app/demo.py [--model meta-llama/Llama-3.1-8B-Instruct]
                       [--adapter experiments/joint/adapter]
                       [--port 7860]

See Section 17–18 of the execution guide for demo design and faculty script.
"""

import argparse
import json
import sys
import secrets
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr

HISTORY_SECRET = secrets.token_hex(16)

# ---------------------------------------------------------------------------
# Preset examples (Section 17: include at least one failure case)
# ---------------------------------------------------------------------------

PRESET_EXAMPLES = [
    # Sarcasm — baseline typically fails, fine-tuned should get right
    {
        "context": "A student completely fails every question on the exam.",
        "utterance": "Wow, you absolutely nailed that.",
        "category": "sarcasm",
        "gold_label": "sarcastic",
        "gold_meaning": "The student performed terribly; the speaker is being sarcastic.",
    },
    # Sarcasm — sincere control (same trigger phrase, different context)
    {
        "context": "A student scores 98% on a very difficult exam.",
        "utterance": "Wow, you absolutely nailed that.",
        "category": "sarcasm",
        "gold_label": "sincere",
        "gold_meaning": "The speaker genuinely congratulates the student on their excellent performance.",
    },
    # Implicature
    {
        "context": "A asks: \"Did you like the movie?\"",
        "utterance": "The popcorn was great.",
        "category": "implicature",
        "gold_label": "no",
        "gold_meaning": "B didn't like the movie — deflecting to the popcorn implies the movie itself wasn't worth commenting on positively.",
    },
    # Indirect speech act
    {
        "context": "Two coworkers in a cold meeting room.",
        "utterance": "It's really cold in here.",
        "category": "indirect_speech_act",
        "gold_label": "request",
        "gold_meaning": "A request to turn up the heat or close the window.",
    },
    # Implicature — harder
    {
        "context": "A asks: \"Is the new intern good at his job?\"",
        "utterance": "He's always on time.",
        "category": "implicature",
        "gold_label": "no",
        "gold_meaning": "The manager is avoiding a direct answer, implying the intern isn't particularly skilled — 'being on time' is faint praise.",
    },
    # Known failure case (honest demo — Section 17)
    {
        "context": "A asks: \"Did you enjoy the concert?\"",
        "utterance": "The seats were comfortable.",
        "category": "implicature",
        "gold_label": "uncertain",
        "gold_meaning": "Ambiguous — could be polite deflection or genuine highlight. Models and humans disagree on this one.",
    },
]


# ---------------------------------------------------------------------------
# Model inference stubs
# ---------------------------------------------------------------------------

# These get replaced with real model inference when the model is loaded
_baseline_model = None
_finetuned_model = None
_tokenizer = None
_local_models = None


def load_models(model_name: str, adapter_path: str | None = None) -> None:
    """Load baseline and fine-tuned models at startup."""
    global _baseline_model, _finetuned_model, _tokenizer

    try:
        from src.evaluation.harness import load_hf_model

        print("Loading baseline model...")
        _baseline_model, _tokenizer = load_hf_model(model_name, adapter_path=None)

        if adapter_path and Path(adapter_path).exists():
            print("Loading fine-tuned model (with adapter)...")
            _finetuned_model, _ = load_hf_model(model_name, adapter_path=adapter_path)
        else:
            print("⚠ No adapter path provided — fine-tuned column will show 'N/A'")
            _finetuned_model = None

    except Exception as e:
        print(f"⚠ Could not load models: {e}")
        raise RuntimeError("Model loading failed. Use --no-model explicitly for the illustrative demo.") from e


def run_inference(context: str, utterance: str, category: str) -> dict:
    """Run inference on both baseline and fine-tuned models."""
    if _local_models is not None:
        entry = _local_models[category]
        prediction = str(entry["present"].predict([context + "\n" + utterance])[0])
        return {"baseline_label": entry["majority"], "finetuned_label": prediction,
                "baseline_raw": "Majority label learned from the training split",
                "finetuned_raw": "Prediction from a trained TF-IDF + logistic regression classifier (not an LLM)"}
    from src.evaluation.prompts import format_prompt, get_valid_labels
    from src.evaluation.harness import generate_hf, extract_label

    valid_labels = get_valid_labels(category)
    template_name = "zero_shot_v1"
    prompt = format_prompt(category, template_name, context, utterance)

    # Baseline prediction
    if _baseline_model is not None and _tokenizer is not None:
        baseline_raw = generate_hf(_baseline_model, _tokenizer, prompt)
        baseline_label = extract_label(baseline_raw, valid_labels)
    else:
        # Check preset match or provide intelligent baseline simulation
        matched_preset = match_preset(context, utterance, category)
        if matched_preset:
            if category == "sarcasm":
                # Baseline LLM often has literal bias on sarcastic items with positive words
                baseline_label = "sincere" if "fails" in context.lower() or "broken" in context.lower() or "terrible" in context.lower() else matched_preset["gold_label"]
            elif category == "implicature":
                # Baseline often treats faint praise or deflection as literal positive
                baseline_label = "yes" if matched_preset["gold_label"] == "no" else matched_preset["gold_label"]
            elif category == "indirect_speech_act":
                # Baseline classifies surface syntax (e.g. statement or question)
                baseline_label = "statement" if "?" not in utterance else "question"
            else:
                baseline_label = matched_preset["gold_label"]
        else:
            if category == "sarcasm":
                baseline_label = "sincere"
            elif category == "implicature":
                baseline_label = "uncertain"
            else:
                baseline_label = "statement" if "?" not in utterance else "question"
        baseline_raw = f"(Demo Mode - Simulated Llama-3.1-8B-Instruct Zero-Shot)\nAnswer: {baseline_label}"

    # Fine-tuned prediction
    if _finetuned_model is not None and _tokenizer is not None:
        finetuned_raw = generate_hf(_finetuned_model, _tokenizer, prompt)
        finetuned_label = extract_label(finetuned_raw, valid_labels)
    elif _baseline_model is not None:
        finetuned_label = "N/A"
        finetuned_raw = "No trained adapter loaded."
    else:
        matched_preset = match_preset(context, utterance, category)
        if matched_preset:
            finetuned_label = matched_preset["gold_label"]
            finetuned_meaning = matched_preset["gold_meaning"]
        else:
            if category == "sarcasm":
                finetuned_label = "sarcastic" if any(w in utterance.lower() for w in ["great", "nailed", "wonderful", "brilliant", "perfect"]) and any(w in context.lower() for w in ["fail", "rain", "break", "bad", "terrible", "late"]) else "sincere"
                finetuned_meaning = "The speaker is using irony given the contradicting context." if finetuned_label == "sarcastic" else "The speaker is stating their genuine perspective."
            elif category == "implicature":
                finetuned_label = "no" if any(w in utterance.lower() for w in ["popcorn", "on time", "busy", "exam", "left", "tried", "slow"]) else "yes"
                finetuned_meaning = "The speaker indirect response implies " + finetuned_label
            else:
                finetuned_label = "request" if any(w in utterance.lower() for w in ["cold", "hot", "stuffy", "salt", "open", "reach", "can you", "could you", "hands are full"]) else "suggestion"
                finetuned_meaning = "The utterance functions as an indirect " + finetuned_label
        finetuned_raw = f"(Demo Mode - Simulated QLoRA Llama-3.1-8B)\nLabel: {finetuned_label}\nIntended meaning: {finetuned_meaning}"

    return {
        "baseline_label": baseline_label,
        "baseline_raw": baseline_raw,
        "finetuned_label": finetuned_label,
        "finetuned_raw": finetuned_raw,
    }


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------


def match_preset(context, utterance, category):
    return next((ex for ex in PRESET_EXAMPLES if ex["context"].strip().casefold() == context.strip().casefold()
                 and ex["utterance"].strip().casefold() == utterance.strip().casefold()
                 and ex["category"] == category), None)


def predict(context: str, utterance: str, category: str) -> str:
    """Main prediction function called by Gradio."""
    if not context or not utterance:
        return "Please provide both context and utterance."

    results = run_inference(context, utterance, category)

    # Find gold answer if this matches a preset example
    gold_label = ""
    gold_meaning = ""
    for ex in PRESET_EXAMPLES:
        if ex == match_preset(context, utterance, category):
            gold_label = ex["gold_label"]
            gold_meaning = ex["gold_meaning"]
            break

    # Format output as a comparison table
    output = f"## Results\n\n"
    if _local_models is not None:
        output += "**Live CPU reference model — majority baseline vs. trained TF-IDF classifier. This is not an LLM comparison.**\n\n"
    elif _baseline_model is None:
        output += "**Illustrative simulation — these outputs are scripted and are not model predictions or research evidence.**\n\n"
    output += f"**Category:** {category}\n\n"
    output += "| | Majority baseline | TF-IDF classifier |\n" if _local_models is not None else "| | Baseline | Fine-Tuned |\n"
    output += f"|---|---|---|\n"
    output += f"| **Predicted Label** | {results['baseline_label']} | {results['finetuned_label']} |\n"

    if gold_label:
        baseline_correct = "✅" if results["baseline_label"] == gold_label else "❌"
        finetuned_correct = "✅" if results["finetuned_label"] == gold_label else "❌"
        output += f"| **Correct?** | {baseline_correct} | {finetuned_correct} |\n"
        output += f"\n**Gold Label:** {gold_label}\n"
        output += f"\n**Reference intended meaning (preset):** {gold_meaning}\n"

    output += f"\n---\n\n**Raw Outputs:**\n"
    baseline_name = "Majority baseline" if _local_models is not None else "Baseline"
    comparison_name = "TF-IDF classifier" if _local_models is not None else "Fine-Tuned"
    output += f"- {baseline_name}: *{results['baseline_raw'][:200]}*\n"
    output += f"- {comparison_name}: *{results['finetuned_raw'][:200]}*\n"

    return output


def load_preset(example_idx: int) -> tuple[str, str, str]:
    """Load a preset example into the input fields."""
    if 0 <= example_idx < len(PRESET_EXAMPLES):
        ex = PRESET_EXAMPLES[example_idx]
        return ex["context"], ex["utterance"], ex["category"]
    return "", "", "sarcasm"


def build_demo() -> gr.Blocks:
    """Build the Gradio demo interface."""
    with gr.Blocks(
        title="Pragmatic Competence in LLMs — Demo",
    ) as demo:
        gr.Button('View graphs and result matrices', link='/graphs', variant='secondary')
        gr.Button('Human interpretation review', link='/review')
        gr.Markdown("**Mode: live CPU reference classifier (TF-IDF, not an LLM).**" if _local_models is not None else "**Mode: illustrative simulation (scripted outputs, no model loaded).**" if _baseline_model is None
                    else "**Mode: live model inference.** " + ("Fine-tuned adapter loaded." if _finetuned_model is not None else "Fine-tuned adapter unavailable."))
        gr.Markdown("# Pragmatic Competence: CPU reference\n\nCompare a majority baseline and a trained lexical classifier. Enter context and utterance or select an example." if _local_models is not None else
            """
            # 🧠 Pragmatic Competence in LLMs
            ### Baseline vs. Fine-Tuned Comparison

            This demo shows how a fine-tuned model interprets pragmatic meaning
            (sarcasm, implicature, indirect speech acts) compared to the baseline.

            **Enter a context and utterance**, or select a preset example below.
            """
        )

        with gr.Row():
            with gr.Column(scale=2):
                context_input = gr.Textbox(
                    label="Context",
                    placeholder="e.g., A student completely fails every question on the exam.",
                    lines=2,
                )
                utterance_input = gr.Textbox(
                    label="Utterance",
                    placeholder='e.g., "Wow, you absolutely nailed that."',
                    lines=1,
                )
                category_input = gr.Dropdown(
                    choices=["sarcasm", "implicature", "indirect_speech_act"],
                    value="sarcasm",
                    label="Pragmatic Category",
                )
                submit_btn = gr.Button("🔍 Analyze", variant="primary")

            with gr.Column(scale=3):
                output = gr.Markdown(label="Results")

        # Preset examples section
        gr.Markdown("### 📋 Preset Examples")
        gr.Markdown("Click an example to load it into the input fields:")

        with gr.Row():
            for i, ex in enumerate(PRESET_EXAMPLES):
                btn = gr.Button(
                    f"{ex['category']}: \"{ex['utterance'][:40]}...\"",
                    size="sm",
                )
                btn.click(
                    fn=lambda idx=i: load_preset(idx),
                    outputs=[context_input, utterance_input, category_input],
                )

        from app.monitor import mount, dashboard, session_plot
        history = gr.BrowserState([], storage_key='pragmatic-label-history', secret=HISTORY_SECRET)

        def analyze_live(context, utterance, category, previous):
            rendered = predict(context, utterance, category)
            updated = list(previous or [])
            # Keep history local to this browser session; never save user inputs.
            if _local_models is not None or _baseline_model is not None:
                for line in rendered.splitlines():
                    if line.startswith('| **Predicted Label**') and line.split('|')[3].strip() != 'N/A':
                        updated.append({'category': category, 'label': line.split('|')[3].strip()})
                        updated = updated[-100:]
                        break
            return rendered, updated

        submit_btn.click(analyze_live, [context_input, utterance_input, category_input, history],
                         [output, history])

    with demo.route('Graphs and matrices', '/graphs'):
        gr.Button('Back to Analyze', link='/')
        gr.Markdown('# Graphs and result matrices\nAll measured research results are on this page. Missing experiments remain explicitly marked.')
        graph_history = gr.BrowserState([], storage_key='pragmatic-label-history', secret=HISTORY_SECRET)
        live_plot = gr.Plot(label='Live input prediction history')
        gr.Markdown('Only category and predicted label are stored in this browser, for the last 100 analyses. History resets when the server restarts. Input text is not stored by this feature.')
        clear = gr.Button('Clear prediction history')
        clear.click(lambda: [], outputs=graph_history)
        graph_history.change(session_plot, graph_history, live_plot)
        demo.load(session_plot, graph_history, live_plot)
        chart_category, chart_outputs = mount(gr)
        demo.load(dashboard, chart_category, chart_outputs)

    with demo.route('Human review', '/review'):
        gr.Button('Back to Analyze', link='/')
        from app.review import mount_review
        mount_review(gr)

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    global _local_models
    parser = argparse.ArgumentParser(description="Launch the pragmatic competence demo.")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Base model name")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to LoRA adapter directory")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port for the Gradio server")
    parser.add_argument("--share", action="store_true",
                        help="Create a public Gradio link")
    parser.add_argument("--no-model", action="store_true",
                        help="Run without loading models (UI-only demo mode)")
    parser.add_argument("--local", action="store_true", help="Use trained CPU reference classifiers")
    parser.add_argument("--open-browser", action="store_true", help="Open the browser after the server is ready")

    args = parser.parse_args()

    if args.local and args.no_model:
        parser.error("Choose either --local or --no-model")
    if args.local:
        import joblib
        _local_models = joblib.load(PROJECT_ROOT / "results/local/classifiers.joblib")
    elif not args.no_model:
        load_models(args.model, args.adapter)

    demo = build_demo()
    demo.launch(server_port=args.port, share=args.share, inbrowser=args.open_browser)


if __name__ == "__main__":
    main()
