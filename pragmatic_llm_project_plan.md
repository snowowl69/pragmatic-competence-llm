# Evaluating and Enhancing Pragmatic Competence in LLMs
### Technical & Research Implementation Blueprint (B.Tech IT Project)

---

## A. Project Definition

**Problem.** LLMs are trained mostly to predict/continue text that is *semantically* coherent. Pragmatics — what a speaker *means* given context, as opposed to what the sentence literally says — is a different and harder skill: it requires modeling the speaker's intent, the listener's assumed knowledge, and the conversational context. Three phenomena where literal and intended meaning diverge:

- **Sarcasm/irony** — literal polarity is flipped by context/tone.
- **Conversational implicature** — the answer is technically off-topic but pragmatically clear (Gricean maxims).
- **Indirect speech acts** — a question is syntactically a question but functions as a request/command.

**Why it matters.** Any system that talks to humans (assistants, moderation, support bots, tutoring systems) will misfire if it takes B's "I have an exam tomorrow" as a literal statement about exams rather than a polite decline. Pragmatic failures produce socially wrong, sometimes harmful outputs (missing a complaint expressed through irony, failing to detect a mitigated disagreement, executing a literal reading of an indirect command).

**Pragmatic competence, computationally defined**, for this project = given `(context, utterance)`, the ability to correctly (1) classify whether/which pragmatic phenomenon is present, and (2) produce the intended meaning, at a rate approaching human agreement, without relying on shallow lexical shortcuts (e.g., "yeah right" ⇒ sarcasm regardless of context).

**System I/O.**
- **Input:** a short dialogue/context + a target utterance (optionally speaker roles/relationship, e.g. "friend", "boss").
- **Output:** (a) pragmatic category label, (b) intended meaning (short generated text or selected option), (c) confidence, (d) for sarcasm specifically, literal-vs-intended polarity.

**What the final system does, concretely:** a benchmark + evaluation harness that scores any LLM on these three phenomena, a LoRA-adapted open model that outperforms its own zero-shot baseline on the benchmark, and a small web demo that shows baseline-vs-fine-tuned side by side on a live example.

---

## B. Research Gap and Novelty

**Already well-covered in prior work** (per the surveys/papers you cited): implicature evaluation on individual benchmarks, sarcasm/irony detection as a standalone classification task, fine-tuning-strategy studies for implicature specifically (Goldilocks paper), human-vs-LLM comparisons on implicature, mitigated-disagreement pragmatics, non-verbal pragmatic inference.

**What's thin:** (1) *unified*, same-format evaluation across sarcasm + implicature + indirect speech acts on the *same* models with the *same* metrics, so cross-category comparisons are apples-to-apples; (2) whether adaptation on one pragmatic phenomenon **transfers** to the others (cross-category generalization) — this is the most under-explored angle and is realistically testable via ablations (Section M); (3) a structured, reusable **failure taxonomy** applied consistently across all three phenomena.

**Honest verdict:** simply "combining three phenomena into one framework" is not sufficient novelty by itself — it's an engineering unification, not a research contribution. The cross-category **transfer/generalization study** is what turns this into a real empirical contribution, because no cited paper directly answers "does implicature fine-tuning help sarcasm detection?"

**Candidate contributions**, ranked (Novelty / Difficulty / Compute / Research value / Feasibility, each 1–5, 5 = highest):

| # | Contribution | Novelty | Difficulty | Compute | Research value | Feasibility |
|---|---|---|---|---|---|---|
| 1 | Unified evaluation harness across 3 phenomena, same models/metrics | 2 | 2 | 1 | 3 | 5 |
| 2 | Cross-category transfer study (train on A, test on B) | 4 | 3 | 2 | 5 | 4 |
| 3 | Structured, consistently-applied failure taxonomy + automatic tagging | 3 | 3 | 1 | 4 | 4 |
| 4 | Context-ablation study (with/without context, with/without literal meaning) | 3 | 2 | 1 | 4 | 5 |
| 5 | Custom pragmatics-aware training objective (auxiliary loss) | 4 | 5 | 4 | 3 | 2 |

**Recommendation:** build 1 (it's your infrastructure either way), and make 2 + 4 the actual research contributions, with 3 as supporting analysis. Skip 5 unless Phase 6–8 finish early (see Section L) — it's high effort for uncertain payoff at this scale.

---

## C. Research Questions & Hypotheses

- **RQ1:** How accurately, and with what error patterns, do current open- and closed-source LLMs interpret sarcasm, conversational implicature, and indirect speech acts, under a common evaluation protocol?
- **RQ2:** Are there systematic, statistically significant performance differences across the three pragmatic categories, and do they correlate with model scale or training regime (base vs. instruction-tuned)?
- **RQ3:** Does parameter-efficient fine-tuning (LoRA) on pragmatic examples from *one* category improve performance on the *other* categories (positive transfer), or does it only help the trained category (or hurt others — catastrophic interference)?
- **RQ4:** How much does explicit context (vs. utterance-only) contribute to correct pragmatic interpretation, and does this contribution change after fine-tuning?
- **RQ5 (optional/stretch):** What error types (from the failure taxonomy) are most reduced by fine-tuning, and which persist regardless of adaptation?

**Hypotheses:**
- H1: Instruction-tuned models will outperform base models of similar size on all three categories, but the *gap between categories* will persist (indirect speech acts easiest, implicature hardest — literal-reading bias is best documented for implicature).
- H2: Fine-tuning on all three categories jointly will generalize better than single-category fine-tuning, but even single-category fine-tuning will show partial positive transfer, because the phenomena share an underlying "look past the literal string" skill.
- H3: Context removal will hurt sarcasm and implicature more than indirect speech acts (indirect requests are often recoverable from the utterance's syntax/politeness markers alone).

---

## D. System Architecture

```
                    ┌─────────────────────────┐
                    │   Dataset Sources        │
                    │ (existing + curated)     │
                    └────────────┬─────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Normalization Layer     │
                    │  → unified JSON schema   │
                    └────────────┬─────────────┘
                                 ▼
                    ┌─────────────────────────┐
                    │  Unified Pragmatic       │
                    │  Benchmark (train/val/test)│
                    └────────────┬─────────────┘
                                 ▼
              ┌──────────────────┴──────────────────┐
              ▼                                      ▼
   ┌─────────────────────┐               ┌─────────────────────┐
   │ Baseline Evaluation  │               │  Training Set Prep   │
   │ (zero/few-shot, all  │               │  (instruction format) │
   │  candidate models)   │               └──────────┬────────────┘
   └──────────┬────────────┘                          ▼
              ▼                             ┌─────────────────────┐
   ┌─────────────────────┐                  │ LoRA / QLoRA Fine-   │
   │ Metrics + Failure    │                  │ Tuning (1 base model)│
   │ Taxonomy Tagging     │                  └──────────┬────────────┘
   └──────────┬────────────┘                             ▼
              │                             ┌─────────────────────┐
              │                             │ Fine-Tuned Model     │
              │                             └──────────┬────────────┘
              │                                         ▼
              │                             ┌─────────────────────┐
              └────────────────────────────▶│ Post-FT Evaluation + │
                                             │ Statistical Compare  │
                                             │ + Ablations (M)      │
                                             └──────────┬────────────┘
                                                          ▼
                                             ┌─────────────────────┐
                                             │  Demo App (Gradio)   │
                                             │  baseline vs FT      │
                                             └─────────────────────┘
```

This keeps your original pipeline but makes explicit that baseline eval and training-data prep branch off the *same* normalized benchmark, and that ablations (context/category transfer) sit alongside — not after — the final comparison, since they're a primary research output, not an afterthought.

---

## E. Dataset & Benchmark Design

**Build vs. combine:** combine existing datasets per phenomenon and normalize into one schema, then hand-curate a small (~150–250 example) contamination-resistant test set yourselves. Building 1,000+ examples from scratch for three phenomena is not realistic for a B.Tech timeline, and existing datasets are exactly what makes the "unified evaluation" contribution possible.

**Candidate public sources** (verify license/availability at project start — search rather than trust memory, since dataset hosting changes):
- Sarcasm/irony: SemEval sarcasm-detection shared tasks, iSarcasm, MUStARD (has dialogue context — valuable), Sarcasm Corpus V2.
- Implicature: the Circa dataset (indirect answers to polar questions), the implicature datasets used in the "Goldilocks" paper, GRICE-style curated implicature sets from recent ACL/EMNLP papers.
- Indirect speech acts: fewer ready-made datasets exist — this category will need the most manual curation; can be partly derived by writing template variants (direct vs. indirect phrasing of the same request) and validating them, or from indirect-request subsets of politeness-strategy corpora.

**Target scale:** ~150–300 examples per category for train, ~50–80 for validation, ~50–100 held-out **hand-verified** test (≈600–1,300 total). This is enough for LoRA fine-tuning signal and statistically usable comparisons, and small enough for 2–3 students to fully quality-check.

**Unified schema** (extends your draft):

```json
{
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
  "requires_context": true,
  "source": "MUStARD",
  "license": "research-use",
  "split": "test",
  "annotator_notes": ""
}
```

`answer_options` lets the same record serve multiple-choice eval; `requires_context` flags examples usable for the context-ablation study; `difficulty` (easy/medium/hard, assigned by whether resolution needs only lexical cues vs. world knowledge) supports failure analysis.

**Quality control:** 2 annotators per example (you two/three, or peer-reviewed within the team) with a third tiebreaker; compute Cohen's κ on a 10% double-annotated sample and report it — this is standard and expected in a pragmatics paper, not optional. Drop or flag examples with annotator disagreement rather than silently keeping them. De-duplicate near-identical utterances. Check your hand-built test set against training data (and, as best you can, against known public pretraining corpora) to reduce contamination risk — this mainly matters for the *test* split, less for train.

**Avoiding lexical-shortcut exploitation:** include *sincere* controls that use surface markers typical of sarcasm ("great job" said genuinely) and *literal* controls for implicature/indirect speech (direct answers/requests that superficially resemble indirect ones), so a model can't just pattern-match trigger words.

---

## F. Benchmark Task Design

Use a **hybrid**: (1) multiple-choice / binary classification as the primary quantitative metric (clean accuracy/F1, no ambiguous scoring), plus (2) short intended-meaning generation as a secondary, qualitative/LLM-judged task for the demo and failure analysis. Pure generation-only benchmarking is hard to score reliably at this scale — classification gives you the clean numbers a research report needs.

**Example items:**

*Sarcasm* — Input: context + utterance. Task: "Is this utterance sincere or sarcastic?" Options: sincere / sarcastic.
*Implicature* — Input: A's question + B's reply. Task: "What is B most likely implying?" Options: yes / no / uncertain (Circa-style), or short-answer generation for open items.
*Indirect speech act* — Input: utterance + context. Task: "What is the speaker's actual intent?" Options: literal question / request / suggestion / complaint.

Model receives a fixed prompt template (Section H) and returns a single label (classification) or a one-sentence paraphrase (generation subset).

---

## G. Models to Evaluate

**Open-source (Hugging Face):**
- **Llama-3.1-8B-Instruct** — good general instruction-following baseline, fits 4-bit QLoRA on a single T4/A10/L4 (Colab/Kaggle-viable). Evaluation + fine-tuning candidate.
- **Mistral-7B-Instruct-v0.3** — comparable size, different training recipe; good for showing results aren't model-specific. Evaluation baseline; fine-tune only if time allows a second run.
- **Phi-3-mini (3.8B)** or **Gemma-2-9B** — smaller/faster alternative if GPU quota is tight; useful as a lightweight fine-tuning candidate.
- **Llama-3.2-1B/3B-Instruct** — fallback if compute is very constrained; expect weaker baseline pragmatic performance, which is itself an interesting data point.

**API/proprietary (evaluation only, not fine-tuned):**
- GPT-4o-mini or GPT-4o, Claude (via API) — used purely as strong upper-bound baselines for comparison; small API cost, budget a fixed number of calls (Section Q covers cost control).

**Recommended primary fine-tuning target: Llama-3.1-8B-Instruct with QLoRA (4-bit).** It's realistic on a single free/Colab-Pro-tier GPU, well-documented, and strong enough zero-shot that improvements are meaningful rather than trivial.

---

## H. Baseline Evaluation Protocol

- **Zero-shot:** task instruction + context + utterance + answer options, temperature 0 (or as close to deterministic as the API/model allows), single pass.
- **Few-shot (3–5 shot):** same prompt with a small fixed set of worked examples *not* drawn from the test set, to see if in-context learning alone closes much of the gap (important control before claiming fine-tuning helped).
- **Prompt sensitivity control:** run at least 2 paraphrased prompt templates per model and report the spread, not just one lucky prompt — pragmatics literature specifically flags prompt sensitivity as a confound.
- **Determinism:** set `temperature=0`, fixed seed where the API allows; for models where 0 isn't fully deterministic, run each item 3x and take majority vote, report variance.

**Results schema:**
```json
{
  "model": "llama-3.1-8b-instruct",
  "setting": "zero-shot",
  "prompt_template": "v1",
  "item_id": "sarc_0001",
  "predicted_label": "sarcastic",
  "gold_label": "sarcastic",
  "correct": true,
  "raw_output": "...",
  "run_index": 0
}
```
Store as JSONL, one row per (model, setting, item, run) — this makes every later metric a simple groupby.

---

## I. Evaluation Metrics

**Necessary:** accuracy, macro-F1 (categories are imbalanced-ish and macro-F1 avoids the majority class dominating), per-category accuracy, confusion matrix per category. **For the paper's core claims, also necessary:** a significance test on baseline-vs-fine-tuned accuracy (McNemar's test is the right choice for paired classification predictions on the same test set — simpler and more appropriate here than a t-test).

**Optional/nice-to-have:** confidence calibration (only if the model exposes token log-probs — not all APIs do), inter-annotator agreement as a human-performance ceiling to compare models against.

**Scoring generated intended-meanings** (secondary task): don't rely on exact string match. Use an LLM-as-judge rubric (0/1/2 scale: wrong / partially captures intent / fully captures intent) scored by a *separate, fixed* judge model, with a human-checked sample (~15–20%) to validate the judge isn't systematically biased — report judge-human agreement, and be explicit in the write-up that LLM-as-judge is a known-imperfect proxy.

---

## J. Failure Taxonomy

Refined categories (merging your list into causes that are actually distinguishable from model output + context):

1. **Literal-bias** — model answers as if only literal semantics matters.
2. **Context under-use** — model ignores available disambiguating context (visible if accuracy drops less than expected when context is removed — see Section M).
3. **World-knowledge gap** — needs outside knowledge (e.g., cultural norm) the model lacks.
4. **Negation/scope error** — mishandles negation or scope within the utterance.
5. **Lexical-shortcut over-reliance** — gets it right on surface-marker-heavy items, wrong on the sincere/literal controls (Section E) — this is a strong, checkable signal.
6. **Ambiguity/no clear gold** — item is genuinely ambiguous even to human annotators (flag from annotation stage, exclude from strict scoring or report separately).
7. **Reasoning-chain failure** — model shows correct reasoning steps (if you elicit a rationale) but a wrong final label, vs. wrong reasoning throughout.

**Categorization method:** automatic tagging where possible (e.g., compare performance on control-vs-marker items for #5, compare context-present-vs-removed items for #2), manual tagging by inspecting a stratified sample (~15 errors per category per model) for the rest. Report as a stacked bar chart (error type × pragmatic category) and a table of example failures with model output.

---

## K. Fine-Tuning Strategy

**Recommendation: QLoRA (4-bit base model + LoRA adapters), supervised instruction fine-tuning.** Full fine-tuning is not realistic on Colab/Kaggle GPUs for a 7–8B model; QLoRA is the standard, well-supported approach for exactly this constraint.

- **Data format:** instruction-style, e.g. `{"instruction": "...task prompt...", "input": "context+utterance", "output": "gold label / short intended-meaning"}`, matching the model's chat template.
- **Train/val/test:** strictly disjoint at the *source-dataset* level too, not just item level, to reduce leakage from near-duplicate items in the same source dataset.
- **Starting hyperparameters** (typical for 7–8B QLoRA on this data scale, adjust empirically): LoRA rank 16 (try 8 and 32 as a small sweep if time allows), LoRA alpha 32, target modules = attention projections (`q_proj`,`k_proj`,`v_proj`,`o_proj`), dropout 0.05, learning rate 1e-4 to 2e-4, 3 epochs, effective batch size 8–16 (via gradient accumulation), 4-bit NF4 quantization (bitsandbytes), bf16 compute dtype.
- **GPU:** single 16GB (T4) is workable at small batch size for 7–8B QLoRA; 24GB (L4/A10) is more comfortable. Expect roughly 30–90 minutes per training run at this dataset scale — budget for several runs (per-category + joint, per Section M).

---

## L. On the "Pragmatics-Aware Fine-Tuning Objective" Claim

**Critical evaluation:** as stated in the abstract, this phrase implies a *modified loss function*, which is a much bigger technical and research commitment than what's actually needed to get a good result. For a B.Tech project, standard supervised instruction fine-tuning **on carefully designed pragmatic supervision** (gold label + short rationale/intended-meaning in the target output) is scientifically defensible and is what "pragmatics-aware" should mean in your abstract — the *data design* is what's pragmatics-aware, not necessarily the loss.

**Minimum viable approach (recommended):** standard cross-entropy LM loss over instruction-tuned examples whose *target output* explicitly encodes pragmatic reasoning (label + one-sentence justification), which is a legitimate and common technique ("chain-of-thought fine-tuning") — no custom loss code needed, fully supported by TRL/PEFT out of the box.

**Advanced research approach (optional, only if MVP finishes early):** an auxiliary loss term that penalizes the model for outputting the *literal* interpretation when the gold is *pragmatic* — e.g. total loss `L = L_LM + λ·L_contrast`, where `L_contrast` is a margin/contrastive term pushing down the model's probability mass on the literal-meaning tokens relative to the intended-meaning tokens for the same input. This requires custom `Trainer` loss overriding in Hugging Face `transformers`/TRL, is meaningfully harder to implement and debug, and its payoff over the MVP approach is uncertain — treat it as a stretch goal, not a plan dependency.

---

## M. Ablation Experiments

Core, must-run set (this is your main research payoff, per Section B):

1. **Single-category fine-tuning → test all categories** (train sarcasm-only, test sarcasm/implicature/indirect; repeat for each category) — directly answers RQ3 (transfer).
2. **Joint fine-tuning (all 3 categories) → test all categories** — the comparison point for #1.
3. **Context-present vs. context-removed** at test time, baseline and fine-tuned — answers RQ4.
4. **Zero-shot vs. few-shot vs. fine-tuned**, same test set — shows fine-tuning gain isn't just something few-shot prompting would've given you for free (important, because reviewers/faculty will ask this).

This is 4 experiment axes, not an explosion of runs — keep it to these; more axes (e.g. also varying LoRA rank across all of them) is where projects blow their timeline. If a rank sweep is wanted, do it only on the joint-fine-tuning run.

---

## N. Demonstration System

**Recommendation: Gradio**, not Streamlit/React/Flask — fastest to build, runs directly in a Colab/HF Space, no separate frontend/backend split needed for a project of this scope, and it's the standard choice for HF-model demos (judges/faculty will recognize the format).

**Improved demo idea:** two text inputs (context, utterance) → on submit, run *both* the baseline and fine-tuned model → display, side-by-side: literal meaning, predicted intended meaning, pragmatic category, confidence (softmax over label options or self-reported), and a ✅/❌ against a small set of pre-loaded example items with known gold answers (so faculty can also try their own text and see it might legitimately fail — be honest about that, don't cherry-pick only wins into the default examples).

**Pipeline:** Gradio frontend → Python backend loads base model + LoRA adapter (via PEFT) once at startup → single inference function called for both model variants → results rendered in a comparison table. Host on a Hugging Face Space (free tier CPU is workable for a 7–8B model only in 4-bit and will be slow; for a snappy demo, host on the same GPU environment you trained in, or use a smaller fine-tuned model as the demo-specific variant if latency matters more than using the exact largest model).

---

## O. Proving the Project Works

Report **per-category accuracy and macro-F1, baseline vs. fine-tuned**, with McNemar's test p-values per category (Section I), plus the transfer-matrix from the ablations (Section M) as the headline result table — that matrix (train-category × test-category accuracy) is the single most convincing artifact for the "does pragmatic learning generalize" question.

**What counts as meaningful improvement:** a statistically significant (p<0.05) gain on the fine-tuned category is the baseline bar to clear. A gain that's numerically large but not significant (common with test sets in the 50–100 range) should be reported as such, not oversold — say "directionally positive but not statistically significant at this sample size" rather than implying a solved problem. If a category *doesn't* improve or gets worse (plausible for indirect speech acts if training skews toward sarcasm/implicature), report that honestly — a "some transfer, some interference" finding is a legitimate and interesting result, not a failure of the project.

---

## P. Real-World Applications

- **Conversational assistants** — *Current failure:* literal execution of indirect requests ("Can you...?" answered as a yes/no capability question). *Why pragmatics matters:* users phrase almost all requests indirectly. *How this helps:* a pragmatics-aware intent layer in front of task execution.
- **Customer support** — *Current failure:* sarcastic/ironic complaints ("great, another broken order") scored as neutral/positive by literal sentiment models. *Why it matters:* misrouted or under-prioritized complaints. *How this helps:* a sarcasm-aware re-scoring step before ticket routing.
- **Content moderation** — *Current failure:* mitigated disagreement/implied hostility slips past literal keyword or toxicity filters. *Why it matters:* real harassment is often indirect. *How this helps:* the failure taxonomy's categories map directly onto moderation edge cases worth flagging for human review.
- **Education** — *Current failure:* automated tutoring systems misread a student's indirect hedge ("I guess that's right?") as confidence rather than uncertainty. *How this helps:* better-calibrated response to implicit uncertainty signals.
- **Social media analysis** — *Current failure:* sarcasm is the classic sentiment-analysis failure mode. *How this helps:* directly what your sarcasm benchmark measures.

**Limitations of deploying your research model directly:** trained on a small, English-only, mostly Western-cultural-context dataset (~1,000 examples); LoRA adapter tuned for a benchmark task format, not open-ended dialogue; no safety/robustness testing for adversarial inputs; not evaluated on real user data distribution. State this explicitly rather than implying production-readiness.

---

## Q. Technology Stack

| Tool | Why |
|---|---|
| Python | project language |
| PyTorch | model loading/training backend |
| Hugging Face `transformers` | load and run open-source LLMs |
| Hugging Face `datasets` | manage/version the benchmark splits |
| `peft` | LoRA/QLoRA adapters |
| `bitsandbytes` | 4-bit quantization for QLoRA on limited VRAM |
| `trl` | supervised fine-tuning trainer (`SFTTrainer`) built for exactly this workflow |
| `scikit-learn` | accuracy/F1/confusion matrix, McNemar's test |
| `pandas` / `numpy` | results wrangling |
| `matplotlib` | result charts (confusion matrices, transfer-matrix heatmap) |
| Jupyter/Colab | iterative development, free/low-cost GPU |
| Git/GitHub | version control, team collaboration |
| Gradio | demo app (Section N) |

Nothing beyond this is needed; resist adding a database, Docker, or a custom web backend — none of it serves the research goal at this scale.

---

## R. Hardware & Software Requirements

**Minimum:** free-tier Colab T4 (16GB VRAM), 12GB system RAM, ~10GB storage (quantized model + adapters + data), Python 3.10+, CUDA 12.x (Colab-managed). Dataset (~600–1,300 examples) is a few MB. Training time: ~30–90 min per QLoRA run on 7–8B at this data scale.

**Recommended:** Colab Pro / Kaggle (30h/week free T4/P100, or A100 quota if available) or university GPU (A10/A100), 16–32GB RAM, 20GB+ storage for multiple LoRA-adapter checkpoints from the ablation runs (Section M needs ~5+ separate fine-tuning runs).

---

## S. Development Phases

| Phase | Objective | Key output | Est. difficulty |
|---|---|---|---|
| 0. Literature review | Confirm gap (Section B), finalize RQs | Related-work summary, finalized RQs | Low |
| 1. Dataset identification | Locate/download source datasets per category | Raw data inventory + license notes | Low–Med |
| 2. Normalization | Convert to unified schema, annotate, QC | `data/processed/*.jsonl`, κ score | Med |
| 3. Benchmark finalization | Build train/val/test splits, contamination check | `data/benchmark/*.jsonl` | Med |
| 4. Baseline evaluation | Run all candidate models zero/few-shot | Raw results JSONL, first metrics table | Med |
| 5. Failure analysis (pre-FT) | Tag errors by taxonomy | Failure charts/tables | Med |
| 6. Fine-tuning | QLoRA runs: per-category + joint | LoRA adapter checkpoints | Med–High |
| 7. Post-FT evaluation | Re-run eval harness on fine-tuned model(s) | Updated metrics table | Low (reuses harness) |
| 8. Ablations | Context on/off, transfer matrix, few-shot control | Transfer-matrix heatmap, significance tests | Med–High |
| 9. Demo | Build Gradio comparison app | Working demo (local or HF Space) | Low–Med |
| 10. Report/paper | Write up | Final report/paper draft | Med |

Common problems to expect: OOM during QLoRA (reduce batch size/increase grad accumulation), API rate limits during baseline eval (add retry/backoff), annotator disagreement eating into timeline (start annotation early, in parallel with Phase 1). Completion criteria for each phase = its output file(s) exist and are reviewed by the whole team before moving on.

---

## T. End-to-End Workflow & Team Division (2–3 students)

**Dependency chain:** Phase 0 → 1 → 2 → 3 (blocking; everything needs the benchmark) → then 4 and "training-data prep" can run in parallel → 5 depends on 4 → 6 depends on 3 (+ can start once training-data format is ready, doesn't need to wait for 4/5) → 7 depends on 6 → 8 depends on 6/7 → 9 depends on 6 (needs a working adapter) but can be scaffolded earlier with dummy outputs → 10 depends on everything.

**Suggested split (3 students):**
- **Student A — Data/Benchmark:** Phases 1–3, plus co-owns annotation QC. Deliverable owner for `data/`.
- **Student B — Evaluation/Analysis:** Phases 4, 5, 8 (ablations/metrics/stats). Deliverable owner for `src/evaluation/`, `src/analysis/`.
- **Student C — Training/Demo:** Phase 6, 7, 9. Deliverable owner for `src/training/`, `app/`.

All three co-write Phase 10 (report), each writing the section matching their phase. For 2 students, merge A+B or B+C depending on strengths (data/annotation-heavy person vs. ML-training-heavy person).

---

## U. Repository Structure

```text
pragmatic-llm/
├── data/
│   ├── raw/              # untouched downloaded source datasets
│   ├── processed/        # normalized, schema-conformant, pre-split
│   └── benchmark/        # final train/val/test JSONL splits
├── src/
│   ├── preprocessing/    # normalization scripts, schema validation, dedup
│   ├── evaluation/       # prompt templates, eval harness, results loader
│   ├── training/         # QLoRA training scripts, configs per run
│   └── analysis/         # metrics, significance tests, failure tagging, plots
├── configs/               # YAML/JSON: LoRA hyperparams, prompt templates, model list
├── notebooks/             # exploratory work only — nothing load-bearing lives only here
├── experiments/           # one subfolder per training/ablation run, with its own results + config snapshot
├── app/                   # Gradio demo
├── tests/                 # schema validation tests, eval-harness sanity tests
├── docs/                  # RQs, taxonomy definitions, annotation guidelines
├── requirements.txt
└── README.md
```

`experiments/` is the important addition to your draft — with 5+ fine-tuning runs (Section M), you need each run's config + results snapshotted separately or you'll lose track of which adapter came from which ablation setting.

---

## V. Feature Scope

**MVP (must work for the project to exist):** normalized benchmark (Section E), baseline zero-shot eval on ≥3 models, one QLoRA fine-tuned model (joint, all categories), post-FT eval, accuracy/F1 comparison table, basic demo.

**Research features (needed for academic credibility):** significance testing (McNemar's), the failure taxonomy applied systematically, the single-category-vs-joint transfer ablation (Section M #1–2), context-ablation (Section M #3).

**Advanced/optional (only if ahead of schedule):** custom pragmatics-contrastive loss (Section L advanced), LoRA-rank sweep, second base model fully fine-tuned for cross-model comparison, LLM-as-judge generation scoring beyond the classification task, deployment as a public HF Space.

---

## W. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Insufficient GPU / Colab disconnects | QLoRA + small models, checkpoint every N steps, resume-from-checkpoint logic |
| Dataset quality/ambiguous items | Double annotation + κ score, explicit "ambiguous" tag excluded from strict scoring |
| Benchmark contamination | Prefer smaller/newer or hand-built test items for the *test* split specifically |
| Overfitting on small fine-tuning set | Held-out val set, early stopping, keep LoRA rank modest |
| Catastrophic forgetting | Evaluate fine-tuned model on a couple of *non*-pragmatic sanity tasks too, report if general ability degraded |
| Unreliable LLM-as-judge | Human-check ~15–20% of judged generation outputs, report judge-human agreement |
| API cost (proprietary baselines) | Fix a hard cap on number of API calls per model up front; cache all responses |
| Reproducibility | Fixed seeds, config files per run, pinned `requirements.txt` |
| Cultural/English-only bias | State explicitly as a limitation; don't claim cross-lingual/cross-cultural generality |
| Model licensing | Confirm each HF model's license permits your use (research/education is usually fine, but check redistribution terms if you publish adapters) |

---

## X. What NOT to Do

- Don't build a custom pragmatics loss function (Section L advanced) as a *dependency* of the core plan — it's a stretch goal only.
- Don't try to build datasets for all three categories entirely from scratch — combine + curate instead (Section E).
- Don't fine-tune more than one primary base model as a core requirement — a second model is a nice-to-have, not MVP.
- Don't claim "first unified framework" or "first to fine-tune for pragmatics" — both are false given the cited prior work; claim the transfer study and unified-protocol comparison instead.
- Don't rely on exact-string-match scoring for generated intended-meanings — it will produce meaningless numbers.
- Don't skip the sincere/literal control items in the dataset — without them you can't distinguish real pragmatic competence from lexical-cue matching, which undermines the entire evaluation's validity.
- Don't run a large hyperparameter grid search across all ablation axes — pick sensible defaults (Section K) and spend the saved time on the transfer study instead.

---

## Y. Final Recommendation

1. **Scope:** unified benchmark (3 phenomena, ~600–1,300 items) + baseline eval across ~4 models (2–3 open, 1 API) + one QLoRA fine-tuned model (Llama-3.1-8B-Instruct) + transfer/context ablations + Gradio demo.
2. **Architecture:** as diagrammed in Section D.
3. **Dataset:** combine MUStARD/iSarcasm (sarcasm), Circa/implicature datasets (implicature), curated + template-derived items (indirect speech acts); normalize to the Section E schema; double-annotate with κ reported.
4. **Models:** Llama-3.1-8B-Instruct (fine-tuning target), Mistral-7B-Instruct (baseline-only comparison), one API model as an upper-bound reference.
5. **Fine-tuning:** QLoRA, standard SFT loss with pragmatic-reasoning-augmented targets (Section L, minimum viable approach).
6. **Evaluation:** accuracy/macro-F1/confusion matrix + McNemar's significance test; LLM-as-judge (human-validated) for the generation subset.
7. **Research contribution:** the cross-category transfer matrix (Section M #1–2) + context-ablation (Section M #3) is the paper's core finding, not just the fine-tuned-vs-baseline gain.
8. **Demo:** Gradio, side-by-side baseline vs. fine-tuned, honest example set (including failures).
9. **Deliverables:** benchmark dataset + schema, eval harness, LoRA adapters + configs, results tables/plots, failure-taxonomy analysis, demo app, review paper + research paper drafts.
10. **Success criteria:** a statistically significant improvement on at least the jointly-fine-tuned categories, a clearly reported transfer matrix (even if the result is "partial transfer" rather than "full transfer" — that's still a real finding), and a reproducible pipeline (fixed configs, seeds, pinned requirements).

**Must Build:** Sections E–I, K (MVP fine-tuning), M items 1–2, N (demo).
**Should Build:** M items 3–4, J (systematic failure taxonomy), full significance testing (I).
**Build Only If Time Allows:** L advanced (custom loss), second fully fine-tuned base model, LoRA-rank sweep, public HF Space deployment.

---

# MASTER IMPLEMENTATION CHECKLIST

**Phase 0 — Literature Review**
- [ ] Re-read the 5 cited papers in full; extract each one's exact evaluated phenomena, datasets, and metrics
- [ ] Confirm gap: no existing paper does the unified 3-phenomenon + cross-category transfer study
- [ ] Finalize RQ1–RQ5 and hypotheses H1–H3 in `docs/research_questions.md`

**Phase 1 — Dataset Identification**
- [ ] Locate and download MUStARD/iSarcasm (or equivalent) for sarcasm
- [ ] Locate and download Circa (or equivalent) for implicature
- [ ] Identify/curate source material for indirect speech acts
- [ ] Record license terms for every source in `data/raw/LICENSES.md`

**Phase 2 — Dataset Normalization**
- [ ] Write `src/preprocessing/normalize.py` converting each source into the unified schema
- [ ] Draft annotation guidelines (`docs/annotation_guidelines.md`)
- [ ] Double-annotate a 10% sample; compute and report Cohen's κ
- [ ] Add sincere/literal control items per category
- [ ] Deduplicate; tag `difficulty` and `requires_context`

**Phase 3 — Benchmark Finalization**
- [ ] Produce train/val/test splits, disjoint at source-dataset level
- [ ] Run a contamination check on the test split
- [ ] Freeze `data/benchmark/*.jsonl` and write `docs/dataset_card.md`

**Phase 4 — Baseline Evaluation**
- [ ] Implement `src/evaluation/harness.py` (prompt templates, zero-shot, few-shot, deterministic settings)
- [ ] Run all candidate models (open + API) on the full test set
- [ ] Store raw results per the Section H schema
- [ ] Compute first accuracy/macro-F1/confusion-matrix pass

**Phase 5 — Pre-FT Failure Analysis**
- [ ] Implement automatic taxonomy tagging (control-item comparison, context-removed comparison)
- [ ] Manually tag a stratified error sample
- [ ] Produce failure-type charts per category/model

**Phase 6 — Fine-Tuning**
- [ ] Build instruction-format training data (per Section K) for: sarcasm-only, implicature-only, indirect-only, joint
- [ ] Set up QLoRA training script (`src/training/train_qlora.py`) with configs in `configs/`
- [ ] Run joint fine-tuning (MVP) — save adapter + config to `experiments/joint/`
- [ ] Run the 3 single-category fine-tunes — save each to `experiments/<category>/`

**Phase 7 — Post-FT Evaluation**
- [ ] Re-run the Phase 4 harness on every fine-tuned adapter
- [ ] Recompute metrics; produce baseline-vs-fine-tuned comparison table

**Phase 8 — Ablations**
- [ ] Build the transfer matrix (train-category × test-category accuracy)
- [ ] Run context-present vs. context-removed comparison
- [ ] Run zero-shot vs. few-shot vs. fine-tuned comparison
- [ ] Run McNemar's test on all key baseline-vs-fine-tuned pairs; record p-values

**Phase 9 — Demo**
- [ ] Build Gradio app loading base model + chosen adapter(s)
- [ ] Wire up side-by-side baseline/fine-tuned comparison view
- [ ] Curate an honest example set (including at least one failure case)

**Phase 10 — Report / Paper**
- [ ] Draft review paper section (survey of prior work, Section B gap analysis)
- [ ] Draft research paper (RQs, method, results tables/plots, transfer-matrix discussion, limitations)
- [ ] Internal team review pass before submission

---
*This document is the approved master specification. On request, it can be converted into a repo-ready `MASTER_PROJECT_PLAN.md` for phase-by-phase implementation by a coding assistant.*
