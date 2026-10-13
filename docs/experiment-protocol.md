# Experiment protocol

## Research targets

Evaluate sarcasm, indirect answers and indirect speech acts using the fixed v2 benchmark. Compare the unadapted model with joint, sarcasm-only, implicature-only and indirect-speech-act adapters. Keep the supplied task category, test IDs, label mapping and inference settings identical for each paired comparison.

## Configured main experiment

| Setting | Value |
|---|---|
| Base model | meta-llama/Llama-3.1-8B-Instruct |
| Quantization | 4-bit NF4 with double quantization |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Target modules | q_proj, k_proj, v_proj, o_proj |
| Learning rate | 0.00015 |
| Epochs | 3 |
| Maximum sequence length | 512 |
| Device batch / gradient accumulation | 2 / 4 |
| Seed | 42 |

Consult the committed YAML configurations for the remaining optimizer and checkpoint settings. Record the actual compute dtype and device from the run; hardware-dependent dtype selection can override an unsuitable configured dtype.

The pipeline runs context-present and context-removed evaluation, few-shot evaluation, and a second prompt template for the baseline and each selected adapter. It then runs paired significance analysis and transfer export. Do not combine prompt variants, repeats or different benchmark versions into a single score. One deterministic run does not establish variability across training seeds.

## Classification and explanations

Report accuracy, macro-F1, confusion matrices, test support and invalid predictions. Missing results are not zero scores. Partial coverage must be labeled provisional. Independently confirm complete benchmark coverage before treating a transfer export as a full experiment.

Classification prompts request labels. The separate interpretation experiment uses `meaning-v1`, greedy decoding and at most 160 generated tokens. It samples up to 100 test items per category with seed 42; the speech-act category contributes all 22 examples. It does not use gold meaning text in prompts. Model, adapter and benchmark identity accompany saved outputs.

Two or more independent reviewers score meaning correctness, use of context and grounding on a 0–2 rubric. Retain sample IDs, reviewer IDs, rationale and timestamps. Report coverage, per-dimension ratings, agreement and unresolved ambiguity; adjudicate disagreements before final interpretation claims. Classification correctness alone does not establish explanation correctness.

## Interpretation limits

The CPU context comparison fits separate classifiers for the two input representations. It is not a fixed-weight LLM context ablation. Public source contamination cannot be ruled out. The small speech-act test set, collapsed Circa labels and unreviewed annotation provenance limit conclusions. Never relabel CPU or synthetic smoke-test results as Llama results.
