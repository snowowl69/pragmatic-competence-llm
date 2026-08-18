# Pragmatic Competence in LLMs — PROJECT EXECUTION GUIDE
### Audited, concrete, build-ready version of the prior blueprint

---

## 0. Audit Note — What's Changing From the Blueprint and Why

Before anything else, here's what I'm correcting from the original blueprint, since you asked me not to just restate it:

1. **Too many fine-tuning runs for the compute/time budget as originally scoped.** The blueprint asked for 4 category-combinations (sarcasm-only, implicature-only, indirect-only, joint) as part of MVP-adjacent work. On a single free/Colab-tier GPU shared across 3 students, that's 4 separate QLoRA runs plus reruns when something breaks. **Correction:** joint fine-tuning is the only MUST DO run; the 3 single-category runs move to SHOULD DO and can be parallelized only if you have 2+ GPU-hours/day available across the team — see Phase 7 and the timeline.
2. **The dataset section named datasets but didn't give a decision table or fallback if a dataset turns out unavailable/restrictively licensed.** Corrected in Section 5 with an explicit table and a Plan B per phenomenon.
3. **No fallback plan for "fine-tuning doesn't help."** This is a real, likely-enough outcome at your data scale that not planning for it is a project-risk, not just a research nuance. Corrected in Section 26 — a null/negative result is scoped as a legitimate, presentable outcome, not a failure state.
4. **The demo in the blueprint was described but not scripted for a faculty review.** Corrected in Section 18 with an actual 5-minute script.
5. **"Indirect speech acts" dataset sourcing was the weakest part of the blueprint and stays weakest here** — flagged explicitly again in Section 5, because it's the one place your team will lose the most time if you don't start it early.
6. Everything else in the blueprint (unified schema, QLoRA choice, Gradio demo, McNemar's test, transfer-matrix as the core contribution) holds up and is kept, now with concrete artifacts attached instead of descriptions.

---

## 1. What We Are Actually Building (Zero-Context Explanation)

**The problem.** When you type text into an LLM, it's very good at understanding what the words literally mean. It's much worse at understanding what you *actually mean*, when those two things differ.

**Semantics vs. pragmatics.** Semantics = the literal, dictionary meaning of a sentence. Pragmatics = the meaning a listener derives once they factor in context, tone, and social convention. "Can you pass the salt?" *semantically* asks about your physical ability to pass salt. *Pragmatically*, everyone understands it's a request.

**Why LLMs struggle.** LLMs are trained mostly to predict plausible next words from huge amounts of text. That teaches them a lot about semantics and even a fair amount about pragmatics implicitly — but they have no guaranteed mechanism for reliably tracking speaker intent, especially when it contradicts the literal words (sarcasm) or requires reading between the lines (implicature).

**The three phenomena, with examples:**

*Sarcasm* — literal meaning is the opposite of intended meaning.
1. Context: it's pouring rain. Utterance: "Great, perfect weather for a picnic." → intended: complaint about the rain.
2. Context: teammate submits code with 40 bugs. Utterance: "Beautiful work as always." → intended: criticism.
3. Context: you wait 2 hours for a doctor. Utterance: "Wow, so punctual." → intended: complaint.

*Conversational implicature* — the reply doesn't literally answer the question, but pragmatically it does.
1. A: "Did you like the movie?" B: "The popcorn was great." → implicature: B didn't like the movie.
2. A: "Is John a good lawyer?" B: "He shows up to court on time." → implicature: not really, that's a weak endorsement.
3. A: "Can I get a raise?" B: "Sales have been slow this quarter." → implicature: no.

*Indirect speech acts* — the sentence is grammatically one type of act (question/statement) but functions as another (request/command).
1. "It's cold in here." → functions as: close the window / turn up the heat.
2. "Do you have a pen?" → functions as: lend me a pen.
3. "I wouldn't sit there if I were you." → functions as: a warning, not just an opinion.

**Why these three specifically:** they cover three distinct *mechanisms* of literal/intended divergence — polarity inversion (sarcasm), relevance-based inference (implicature), and speech-act reclassification (indirect requests) — so a model that handles all three plausibly has a general "look past literal meaning" skill, not just a trick for one phenomenon.

**The benchmark** is a set of labeled examples (context + utterance + correct interpretation) used to score any model's pragmatic understanding, the same way a spelling test scores spelling.

**Fine-tuning** means further training an already-trained model on your specific examples so it gets better at your specific task.

**QLoRA** is a fine-tuning technique that (a) shrinks the base model's memory footprint via 4-bit quantization, then (b) trains only a small set of added "adapter" parameters instead of the whole model — this is what makes fine-tuning an 8-billion-parameter model possible on a free/single Colab GPU instead of needing a data-center.

**The final application** is a small web tool: you type in a context and an utterance, and it shows you what the original (baseline) model thought it meant vs. what your fine-tuned model thinks it meant, next to the correct answer.

**The whole project in one sentence:** *We're testing whether AI language models can tell what people actually mean instead of just what they literally said, and whether we can teach a model to get better at that skill in a way that generalizes across different kinds of "meaning more than you say."*

---

## 2. Final Deliverables

**Research Deliverables** — *why each exists*
- **Benchmark** (proof you can measure the problem at all)
- **Evaluation framework** (proof your measurement is fair/reproducible across models)
- **Baseline results** (establishes the problem is real and quantifies it — your "before" picture)
- **Failure analysis** (explains *why* models fail, not just *that* they fail — this is what makes it research, not a leaderboard)
- **Fine-tuned model results** (your intervention's effect — the "after" picture)
- **Transfer experiments** (your central research contribution — see Section 15)
- **Statistical analysis** (proves any observed difference isn't noise)
- **Written research findings** (the actual claims your project can defend)

**Software Deliverables** — *why each exists*
- **Preprocessing pipeline** (turns messy heterogeneous datasets into one usable format — without it nothing downstream works)
- **Evaluation pipeline** (reusable harness — you'll run it 10+ times across baseline/post-FT/ablations, so it must not be a one-off script)
- **Training pipeline** (reproducible QLoRA runs, not manual notebook edits every time)
- **Model/adapters** (the actual artifact you fine-tuned — small files, easy to store/share)
- **Experiment configs** (so every run is reproducible and comparable — this is what stops your ablation results from being "we think we changed one thing")
- **Visualization/reporting scripts** (turns raw JSONL into the tables/charts your report and demo need)
- **Demo application** (the thing faculty actually interacts with — see Section 17)

**Documentation Deliverables** — *why each exists*
- **README** (onboarding for anyone opening the repo, including your future selves)
- **Dataset documentation** (source, license, schema, annotation process — required for academic defensibility and reuse)
- **Methodology doc** (what you did and why, independent of results)
- **Experiment documentation** (per-run configs/results, so you can explain any number in the final report on demand)
- **Final report** (your semester deliverable)
- **Research paper draft** (the review + research paper your faculty project requires)

---

## 3. End-to-End Workflow

```
Literature review → Research gap confirmation
        ↓
Dataset collection (parallel: sarcasm / implicature / indirect)
        ↓
Dataset cleaning + normalization → unified schema
        ↓
Annotation + QC (κ scoring, control items)
        ↓
Benchmark frozen (train/val/test)
        ↓                    ↘
Baseline model evaluation      Training-data formatting
        ↓                            ↓
Failure analysis (pre-FT)      QLoRA fine-tuning (joint, MUST; per-category, SHOULD)
        ↓                            ↓
        └───────────► Post-FT evaluation ◄───────────┘
                            ↓
        Transfer matrix + context ablation + significance testing
                            ↓
                    Demo application
                            ↓
              Final results consolidation
                            ↓
                    Report + research paper
```

Per-stage breakdown:

| Stage | What / Why | In → Out | Owner (of 3) | Gate to next stage |
|---|---|---|---|---|
| Lit review | Confirm gap is real | papers → `docs/research_questions.md` | All | RQs signed off by whole team |
| Dataset collection | Get raw source data | dataset search → `data/raw/` | Student A | Licenses confirmed |
| Cleaning/normalization | One schema for everything | raw → `data/processed/` | Student A | Schema validated (Section 6) |
| Annotation/QC | Trustworthy labels | processed → labeled+κ | Student A (+ all for double-annotation) | κ ≥ 0.6 on sample, or flagged items excluded |
| Benchmark freeze | Fixed splits | labeled → `data/benchmark/*.jsonl` | Student A | No source-level leakage across splits |
| Baseline eval | Quantify the problem | benchmark + models → results JSONL | Student B | All models × all items have a result row |
| Pre-FT failure analysis | Explain *why* | results → taxonomy tags + charts | Student B | Taxonomy applied to ≥15 errors/category/model |
| Training data prep | Feed the fine-tune | benchmark train split → instruction JSONL | Student C | Format matches base model's chat template |
| QLoRA fine-tuning | The intervention | training data → adapter checkpoint | Student C | Training loss converges, val loss doesn't diverge |
| Post-FT eval | Measure the effect | adapter + harness → results JSONL | Student B/C | Same harness as baseline, same test set |
| Transfer + ablation + stats | The research payoff | all results → transfer matrix, p-values | Student B | McNemar's run on every key comparison |
| Demo | Show it | adapter + baseline model → Gradio app | Student C | Runs end-to-end on a fresh restart |
| Report/paper | Communicate it | everything → final docs | All | Every claimed number traceable to a results file |

---

## 4. Phase-by-Phase Implementation Plan

### Phase 0 — Literature Review & Gap Confirmation — **MUST DO**
- **Objective:** lock in the research gap so the rest of the project has a target.
- **Tasks:** re-read the 5 cited papers; write a 1-page gap statement; finalize RQ1–RQ5.
- **Tech:** none (reading/writing).
- **In:** the 5 papers + any others found. **Out:** `docs/research_questions.md`, `docs/related_work.md`.
- **Files to create:** `docs/research_questions.md`, `docs/related_work.md`.
- **Time:** 1 week. **Difficulty:** Low.
- **Dependencies:** none.
- **Completion criteria:** all 3 students can state the gap and RQs in one sentence each without notes.
- **Common problems:** scope creep (finding "one more paper" indefinitely) — timebox it.
- **Show your guide:** the gap statement + finalized RQs, for sign-off before touching data.

### Phase 1 — Dataset Collection — **MUST DO**
- **Objective:** get raw source data for all 3 phenomena.
- **Tasks:** download MUStARD + iSarcasm (sarcasm); download Circa (implicature); collect/curate indirect-speech-act examples (weakest source — start this early, see Section 5).
- **Tech:** Hugging Face `datasets`, manual search/download.
- **In:** dataset repositories. **Out:** `data/raw/`.
- **Files:** `src/preprocessing/download.py`, `data/raw/LICENSES.md`.
- **Time:** 1–2 weeks (indirect speech acts will eat most of this). **Difficulty:** Low–Medium.
- **Dependencies:** Phase 0 (to know exactly what you need).
- **Completion criteria:** raw files present + license terms logged for each source.
- **Common problems:** dataset link dead/paywalled/license unclear — verify at download time (don't trust source lists from memory, including this guide's, without checking); indirect-speech-act sourcing takes longer than expected — budget accordingly.
- **Show your guide:** the dataset table (Section 5) with your actual final choices.

### Phase 2 — Cleaning & Normalization — **MUST DO**
- **Objective:** one consistent schema across all sources.
- **Tasks:** write per-source parsers, map into the unified schema (Section 6), dedupe.
- **Tech:** pandas, `src/preprocessing/normalize.py`.
- **In:** `data/raw/`. **Out:** `data/processed/*.jsonl`.
- **Files:** `src/preprocessing/normalize.py`, `src/preprocessing/schema.py` (validator).
- **Time:** 1 week. **Difficulty:** Medium.
- **Dependencies:** Phase 1.
- **Completion criteria:** every record passes schema validation; no exact duplicates.
- **Common problems:** inconsistent context formatting across sources (some have multi-turn context, some don't) — normalize context to a single string field regardless of source turn structure.
- **Show your guide:** 5 example normalized records per phenomenon.

### Phase 3 — Annotation & QC — **MUST DO**
- **Objective:** trustworthy, contamination-checked labels.
- **Tasks:** write annotation guidelines; double-annotate a 10–15% sample; compute Cohen's κ; add sincere/literal control items; tag `difficulty`/`requires_context`.
- **Tech:** spreadsheet or simple annotation script; `scikit-learn` (or `statsmodels`) for κ.
- **In:** `data/processed/`. **Out:** labeled + QC'd data.
- **Files:** `docs/annotation_guidelines.md`, `src/analysis/agreement.py`.
- **Time:** 1–2 weeks (run in parallel with late Phase 2). **Difficulty:** Medium.
- **Dependencies:** Phase 2.
- **Completion criteria:** κ reported; disagreements resolved or flagged `ambiguous`.
- **Common problems:** annotators drift from the guideline over time — do a calibration pass on the first 10 items together before splitting up the rest.
- **Show your guide:** κ score + 3 example disagreement resolutions.

### Phase 4 — Benchmark Freeze — **MUST DO**
- **Objective:** final, fixed train/val/test splits.
- **Tasks:** split at the *source-dataset* level to avoid leakage; run a basic contamination check on the test split; write the dataset card.
- **Tech:** pandas.
- **In:** QC'd data. **Out:** `data/benchmark/{train,val,test}.jsonl`.
- **Files:** `src/preprocessing/split.py`, `docs/dataset_card.md`.
- **Time:** 2–3 days. **Difficulty:** Low–Medium.
- **Dependencies:** Phase 3.
- **Completion criteria:** splits are frozen and checksummed; nobody edits them after this point without a version bump.
- **Common problems:** temptation to "just fix one more label" mid-experiment — don't; log it as a known-issue instead and fix in a versioned update.
- **Show your guide:** final split sizes per category, dataset card.

### Phase 5 — Baseline Evaluation — **MUST DO**
- **Objective:** quantify current LLM pragmatic performance.
- **Tasks:** implement the eval harness (Section 10); run zero-shot + few-shot on all chosen models; store raw results.
- **Tech:** `transformers`, model APIs, `pandas`.
- **In:** `data/benchmark/test.jsonl` + models. **Out:** `results/baseline/*.jsonl`.
- **Files:** `src/evaluation/harness.py`, `configs/prompts.yaml`.
- **Time:** 1 week. **Difficulty:** Medium.
- **Dependencies:** Phase 4.
- **Completion criteria:** every (model × setting × item) combination has a stored result.
- **Common problems:** API rate limits (add retry/backoff), non-deterministic outputs even at temperature 0 for some APIs (run 3x, majority vote, report variance).
- **Show your guide:** first baseline accuracy table.

### Phase 6 — Pre-FT Failure Analysis — **SHOULD DO**
- **Objective:** explain *why* baseline models fail.
- **Tasks:** auto-tag failures using control-item and context-removed comparisons; manually tag a stratified sample; produce charts.
- **Tech:** pandas, matplotlib.
- **In:** baseline results. **Out:** `results/failure_analysis/`.
- **Files:** `src/analysis/failure_tagging.py`.
- **Time:** 3–5 days. **Difficulty:** Medium.
- **Dependencies:** Phase 5.
- **Completion criteria:** taxonomy (Section 12) applied to ≥15 errors per category per model.
- **Common problems:** taxonomy categories overlapping/ambiguous on real examples — allow multi-tagging rather than forcing one label.
- **Show your guide:** failure-type distribution chart.

### Phase 7 — Fine-Tuning — **MUST DO (joint) / SHOULD DO (per-category)**
- **Objective:** produce the intervention model(s).
- **Tasks:** format training data as instructions; configure QLoRA (Section 13); run joint fine-tuning (MUST); run per-category fine-tunes if GPU time allows (SHOULD).
- **Tech:** `peft`, `bitsandbytes`, `trl`.
- **In:** `data/benchmark/train.jsonl`. **Out:** LoRA adapter checkpoints in `experiments/<run>/`.
- **Files:** `src/training/train_qlora.py`, `configs/lora_joint.yaml`, `configs/lora_<category>.yaml`.
- **Time:** joint run 1–2 days incl. debugging; each additional per-category run ~half a day. **Difficulty:** Medium–High.
- **Dependencies:** Phase 4 (training data derives from the frozen train split).
- **Completion criteria:** training loss decreases and validation loss doesn't diverge; adapter loads and produces sane outputs on a few manual test prompts.
- **Common problems:** OOM (reduce batch size, increase grad accumulation), Colab disconnects mid-run (checkpoint every N steps, resume logic).
- **Show your guide:** training loss curve, 3 example before/after model outputs.

### Phase 8 — Post-FT Evaluation — **MUST DO**
- **Objective:** measure the intervention's effect, apples-to-apples with baseline.
- **Tasks:** rerun the *exact* Phase 5 harness on each fine-tuned adapter.
- **Tech:** same as Phase 5.
- **In:** adapters + `data/benchmark/test.jsonl`. **Out:** `results/post_ft/*.jsonl`.
- **Files:** none new — reuse `src/evaluation/harness.py`.
- **Time:** 2–3 days. **Difficulty:** Low (infrastructure reuse).
- **Dependencies:** Phase 7.
- **Completion criteria:** results directly comparable row-for-row with baseline (same item IDs, same prompt templates).
- **Common problems:** accidentally changing the prompt template between baseline and post-FT runs, which invalidates the comparison — diff the configs before running.
- **Show your guide:** baseline-vs-fine-tuned accuracy table.

### Phase 9 — Transfer Matrix, Context Ablation, Statistics — **MUST DO**
- **Objective:** the actual research contribution.
- **Tasks:** build the transfer matrix (Section 15); run context-present-vs-removed (Section 16); run McNemar's test on all key pairs.
- **Tech:** pandas, `scipy.stats` (McNemar's), matplotlib (heatmap).
- **In:** all results so far. **Out:** `results/transfer_matrix.csv`, `results/context_ablation.csv`, p-values.
- **Files:** `src/analysis/transfer.py`, `src/analysis/significance.py`.
- **Time:** 1 week. **Difficulty:** Medium–High.
- **Dependencies:** Phase 8 (needs per-category adapters for the full matrix — if only the joint adapter exists, the matrix degrades to a single row, which is a legitimate fallback, see Section 26).
- **Completion criteria:** every cell in the transfer matrix has a value; every headline claim has a p-value attached.
- **Common problems:** small test-set sizes making significance hard to reach — report honestly (Section 11), don't drop the test.
- **Show your guide:** the transfer-matrix heatmap — this is your single best result-review slide.

### Phase 10 — Demo Application — **MUST DO**
- **Objective:** a working, faculty-facing demonstration.
- **Tasks:** build the Gradio app (Section 17); curate an honest example set.
- **Tech:** Gradio, `peft` (to load adapter at inference).
- **In:** base model + adapter. **Out:** working `app/demo.py`.
- **Time:** 3–5 days. **Difficulty:** Low–Medium.
- **Dependencies:** Phase 7 (needs at least the joint adapter).
- **Completion criteria:** runs end-to-end from a clean environment restart, not just in the dev notebook.
- **Common problems:** demo works in the training notebook but breaks in a fresh session (missing dependency pins) — test in a clean Colab runtime before review day.
- **Show your guide:** live run-through, see Section 18 script.

### Phase 11 — Report & Research Paper — **MUST DO**
- **Objective:** communicate everything.
- **Tasks:** write review paper (related work + gap), research paper (method/results/discussion), final report.
- **Tech:** LaTeX/Word, your own results.
- **In:** everything. **Out:** final documents.
- **Time:** 2 weeks + buffer. **Difficulty:** Medium.
- **Dependencies:** Phase 9 (need final numbers).
- **Completion criteria:** every number in the paper traces to a specific results file.
- **Common problems:** starting the write-up too late — start the methodology section as soon as Phase 4 freezes, don't wait for all results.
- **Show your guide:** full draft for review before submission deadline, not the day of.

---

## 5. Dataset Strategy

| Dataset | Phenomenon | Size (approx.) | Context? | Labels | License | Use |
|---|---|---:|---|---|---|---|
| MUStARD | Sarcasm | ~690 utterances | Yes (dialogue) | sarcasm/not | Research use (verify current terms) | Train + test |
| iSarcasm | Sarcasm | ~4,400 tweets | Limited | sarcasm/not + irony type | Research use (verify) | Train (supplement) |
| Circa | Implicature | ~34,000 QA pairs | Yes (scenario) | yes/no/other indirect-answer categories | Apache 2.0-style (verify) | Train + test (subsample) |
| "Goldilocks" paper implicature set | Implicature | Varies by release | Yes | implicature-resolution labels | Check paper's release terms | Supplement/test |
| Indirect-request corpora / hand-curated | Indirect speech acts | To be built, target ~150–250 | Yes (you write it) | direct/indirect + intended act | Your own (CC-BY on your own additions) | Train + test |

**A/B/C/D recommendation: B + C, primarily.** Combine existing datasets for sarcasm and implicature (B) — don't rebuild what already exists and is annotated to a reasonable standard. For indirect speech acts, you'll need to create additional examples yourselves (C), since no dataset here is as clean as MUStARD/Circa — budget real annotation time for this, starting in Phase 1, not Phase 2.

**Also do D** — reserve a small (~50–100 item), hand-verified, cross-checked-against-training-sources test set, specifically to reduce contamination risk, even though your training data comes from B/C. This is the set your headline numbers should ultimately be reported against.

**Plan B if a listed dataset is unavailable/relicensed by the time you check (verify at download time — dataset hosting changes):** for sarcasm, fall back to any current SemEval sarcasm-detection shared-task release; for implicature, fall back to hand-curating implicature items using the same template method you're already using for indirect speech acts, sized down accordingly (~100–150 items is still usable for QLoRA at this scale).

---

## 6. Benchmark Design

**Final schema** (same as the blueprint's, with `answer_options` and QC fields kept):

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

**5 sample records:**

```json
{"id": "sarc_0002", "context": "It has been raining nonstop for a week.", "speaker": "colleague", "utterance": "Perfect weather for the company picnic.", "literal_meaning": "The weather is good for the picnic.", "intended_meaning": "The speaker is complaining that the weather is bad.", "pragmatic_category": "sarcasm", "label": "sarcastic", "answer_options": ["sincere", "sarcastic"], "difficulty": "easy", "requires_context": true, "source": "MUStARD", "split": "train"}

{"id": "impl_0002", "context": "A: 'Is the new intern good at his job?'", "speaker": "manager", "utterance": "He's always on time.", "literal_meaning": "The intern arrives on time.", "intended_meaning": "The manager is avoiding a direct answer, implying the intern isn't particularly skilled.", "pragmatic_category": "implicature", "label": "no", "answer_options": ["yes", "no", "uncertain"], "difficulty": "medium", "requires_context": true, "source": "Circa-style", "split": "train"}

{"id": "ind_0002", "context": "Two coworkers in a meeting room.", "speaker": "coworker", "utterance": "It's really stuffy in here.", "literal_meaning": "A statement about the air quality of the room.", "intended_meaning": "A request to open a window or turn on the AC.", "pragmatic_category": "indirect_speech_act", "label": "request", "answer_options": ["statement", "request", "question", "complaint"], "difficulty": "easy", "requires_context": false, "source": "hand-curated", "split": "train"}

{"id": "ctrl_0001", "context": "A student scores 98% on the exam.", "speaker": "friend", "utterance": "Wow, you absolutely nailed that.", "literal_meaning": "The student performed very well.", "intended_meaning": "The speaker sincerely means the student did well.", "pragmatic_category": "sarcasm", "label": "sincere", "answer_options": ["sincere", "sarcastic"], "difficulty": "easy", "requires_context": true, "source": "control-item", "split": "test", "annotator_notes": "sincere control using a marker phrase identical to a common sarcasm trigger — tests lexical-shortcut reliance"}

{"id": "hard_0001", "context": "A: 'Did you enjoy the concert?'", "speaker": "friend", "utterance": "The seats were comfortable.", "literal_meaning": "A statement about seat comfort.", "intended_meaning": "Ambiguous — could be a polite deflection (implying the concert itself wasn't great) or genuinely mean the seats were the highlight; annotators disagreed.", "pragmatic_category": "implicature", "label": "ambiguous", "answer_options": ["yes", "no", "ambiguous"], "difficulty": "hard", "requires_context": true, "source": "hand-curated", "split": "test", "annotator_notes": "flagged — excluded from strict accuracy scoring, reported separately"}
```

**Preventing lexical-shortcut memorization:** the control item above (`ctrl_0001`) reuses the exact sarcasm-trigger phrase ("Wow, you absolutely nailed that") in a context where it's *sincere*. Include one such control per common surface marker per category. Score models separately on marker-items vs. control-items — a large accuracy gap between the two is direct, reportable evidence of shortcut reliance (this feeds directly into the failure taxonomy, Section 12, category "lexical-shortcut over-reliance").

---

## 7. Annotation Process

**Workflow:**
1. Write `docs/annotation_guidelines.md` with 1 worked example per category per label, before annotating anything for real.
2. Calibration round: all annotators independently label the same 10 items, compare, resolve disagreements together, refine the guideline if needed.
3. Split remaining items across annotators, but double-annotate a fixed 10–15% random sample across the whole set (not just the calibration round).
4. Compute Cohen's κ on the double-annotated sample per category. κ ≥ 0.6 is a reasonable "trustworthy" bar for this kind of subjective task; below that, revisit the guideline before proceeding, don't just proceed anyway.
5. For disagreements outside the calibration round: a third team member acts as tiebreaker; if still unresolved, tag `difficulty: hard` / `label: ambiguous` and exclude from strict-scoring but keep in the dataset (reported separately, per Section 6's `hard_0001` example).
6. Deduplicate near-identical items (same utterance, trivially reworded context) before freezing splits.
7. Check for train/test leakage at the *source-dataset* level (an item and its near-duplicate shouldn't end up in different splits) — do this as a script-based check (fuzzy string match on utterance text), not manually.
8. Contamination note: for the hand-curated test subset, this isn't really a risk (you wrote it); for test items drawn from public datasets, you can't fully rule out a model having seen them during pretraining — mitigate by weighting your headline claims more heavily on the hand-curated portion of the test set, and say so explicitly in the report.

---

## 8. Model Selection

**Baseline evaluation models (3, not 5 — trim the blueprint's list):**
- **Llama-3.1-8B-Instruct** — strong open instruction model, realistic on Colab/Kaggle in 4-bit; also your fine-tuning target, so its baseline is the direct "before" picture.
- **Mistral-7B-Instruct** (latest available minor version at the time you run) — comparable size, different training lineage, shows results aren't Llama-specific.
- **One API model** (e.g. a current mid-tier GPT or Claude model) — as an upper-bound reference point only; verify current model names/pricing at the time you actually run this, since offerings change.

**Fine-tuning model: Llama-3.1-8B-Instruct, single choice.** Don't fine-tune a second base model as MUST DO — it doubles GPU time for a comparison that isn't central to your RQs. If time allows, revisit as OPTIONAL.

**Hardware/cost per model:**
- Llama-3.1-8B / Mistral-7B: ~16GB VRAM in 4-bit for inference, fits free-tier Colab T4; fine-tunable via QLoRA on the same hardware.
- API model: no local hardware, but has per-call cost — fix a hard cap (e.g., exactly one pass over the test set per setting, cached, no reruns) before you start.

**Is this appropriate for a student project?** Yes — this is the realistic, well-documented range other university-scale pragmatics/fine-tuning projects use; going bigger (e.g., 70B models) adds cost without adding to what your RQs actually need.

---

## 9. Technology Stack

| Tool | Necessary? | Use |
|---|---|---|
| Python 3.10+ | Yes | project language |
| PyTorch | Yes | model backend |
| `transformers` | Yes | load/run models |
| `datasets` | Yes | manage benchmark splits |
| `peft` | Yes | LoRA/QLoRA adapters |
| `bitsandbytes` | Yes | 4-bit quantization |
| `trl` | Yes | `SFTTrainer` for fine-tuning |
| `scikit-learn` | Yes | accuracy/F1/confusion matrix |
| `scipy` | Yes | McNemar's test (`scipy.stats`) |
| `pandas` / `numpy` | Yes | data wrangling |
| `matplotlib` | Yes | charts |
| Jupyter/Colab | Yes | dev environment, free GPU |
| Git/GitHub | Yes | version control |
| Gradio | Yes | demo |

Nothing else — no database, no Docker, no separate backend framework (Flask/FastAPI unnecessary given Gradio covers the demo needs), no cloud infra beyond Colab/Kaggle.

**Environment setup:** Python 3.10 or 3.11; `pip install torch transformers datasets peft bitsandbytes trl scikit-learn scipy pandas numpy matplotlib gradio`, pinned versions in `requirements.txt` once you've confirmed a working combination (bitsandbytes/CUDA compatibility is the most common breakage point — pin exact versions once it works). GPU: 16GB VRAM minimum (T4), 24GB recommended (L4/A10) if available via Colab Pro or university resources. Storage: ~15–20GB (quantized model, multiple adapter checkpoints, dataset).

---

## 10. Baseline Evaluation — Exact Workflow

```
benchmark item → prompt template (fills context/utterance/options)
              → model (temperature=0, fixed seed where possible)
              → raw text response
              → extraction step (regex/exact-match on option keyword, or a
                 constrained-decoding / forced-choice setup if the model API
                 supports it)
              → compared against gold label
              → logged as one row (Section H schema from the blueprint)
```

**Example prompt (sarcasm, zero-shot):**
```
You will be given a short context and an utterance. Decide whether the
utterance is SINCERE or SARCASTIC given the context. Respond with exactly
one word: "sincere" or "sarcastic".

Context: {context}
Utterance: "{utterance}"
Answer:
```

**Example prompt (implicature):**
```
You will be given a short exchange. Based on B's reply, decide what B is
implying is the answer to A's question: "yes", "no", or "uncertain".
Respond with exactly one word.

A: {question}
B: "{reply}"
Answer:
```

**Avoiding prompt-favoritism between models:** use the *same* template text across all models for a given category (only the model changes); additionally run a second, differently-worded template on all models and report the spread rather than picking whichever template flatters your favorite model. Never hand-tune a prompt per model — that's the single easiest way to accidentally bias a comparison.

**Multiple runs:** for models where temperature=0 doesn't guarantee determinism (some hosted APIs), run each item 3 times, take majority vote as the prediction, and separately report how often the 3 runs disagreed (a stability metric).

---

## 11. Evaluation Metrics

**Primary (must appear in the final presentation):** accuracy, macro-F1, per-category accuracy, confusion matrix, McNemar's test p-value on baseline-vs-fine-tuned pairs.

**Secondary:** LLM-as-judge score (0/1/2 scale) for the generation subset, human-judge agreement rate on a validation sample of the judge's scores.

**Optional:** confidence calibration (only if log-probs are available), a stability metric from repeated runs (Section 10).

**Example final results table:**

| Model | Setting | Sarcasm Acc | Implicature Acc | Indirect Acc | Macro-F1 | vs. baseline p-value |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B | Zero-shot baseline | 0.64 | 0.58 | 0.71 | 0.64 | — |
| Llama-3.1-8B | Few-shot baseline | 0.69 | 0.62 | 0.74 | 0.68 | — |
| Llama-3.1-8B | Fine-tuned (joint) | 0.78 | 0.70 | 0.80 | 0.76 | 0.03 (sarcasm), 0.08 (implicature), 0.21 (indirect) |
| Mistral-7B | Zero-shot baseline | 0.61 | 0.55 | 0.69 | 0.62 | — |
| GPT-4o-mini (ref.) | Zero-shot | 0.79 | 0.73 | 0.85 | 0.79 | — |

(Numbers above are illustrative placeholders to show the table shape — not predictions of your actual results.)

---

## 12. Failure Taxonomy — Practical Version

| Type | Definition | Example | Auto-detectable? | Manual needed? |
|---|---|---|---|---|
| Literal-bias | Answers as if only literal meaning exists | Sarcasm labeled "sincere" when clearly ironic | Partial (compare error rate on marker-heavy items) | Yes, to confirm |
| Context under-use | Ignores available disambiguating context | Same wrong answer with/without context provided | Yes (compare context-on vs context-off predictions) | No |
| World-knowledge gap | Needs outside knowledge the model lacks | Missing cultural reference needed to read the tone | No | Yes |
| Lexical-shortcut | Relies on surface trigger words | Correct on marker items, wrong on matched controls | Yes (control-item accuracy gap) | No |
| Negation/scope error | Mishandles "not", "never", scope of negation | Misreads "I wouldn't say he's bad" | Partial (flag items containing negation, compare accuracy) | Yes |
| Ambiguity / no clear gold | Genuinely unclear even to humans | Annotator disagreement item | No (comes from annotation stage) | N/A — pre-flagged |
| Reasoning-chain failure | Right reasoning steps, wrong final label (or vice versa) | Elicited rationale correctly identifies sarcasm but final label is wrong | No | Yes |

**How many to manually inspect:** ~15 errors per category per model (≈135 total across 3 categories × 3 models) is a defensible, achievable sample for a B.Tech team — report the taxonomy distribution as a stacked bar chart plus 3–5 illustrative example errors with the model's actual output shown.

---

## 13. Fine-Tuning — Beginner-Friendly then Concrete

**Concepts, briefly:** supervised fine-tuning = further training on labeled input→output pairs. LoRA = instead of updating all of a model's billions of weights, you freeze them and train small added "adapter" matrices that get combined with the frozen weights at inference — far fewer trainable parameters, far less memory. QLoRA = LoRA on top of a 4-bit quantized (compressed) copy of the base model, cutting memory further, which is what makes this possible on a single consumer/Colab GPU. You're not training the whole model because (a) you don't have the compute, and (b) it isn't necessary — the adapter captures enough task-specific adjustment for a project at this data scale.

**Implementation plan:**
```
train.jsonl → format as instruction/response pairs matching the model's
              chat template
           → tokenize
           → load base model in 4-bit (bitsandbytes)
           → attach LoRA adapter (peft) targeting attention projections
           → train (trl SFTTrainer), monitor train + val loss
           → save adapter checkpoint
           → reload base model + adapter for evaluation (Phase 8)
```

**Realistic starting hyperparameters:** LoRA rank 16, alpha 32, target modules `q_proj,k_proj,v_proj,o_proj`, dropout 0.05, learning rate 1e-4–2e-4, 3 epochs, effective batch size 8–16 via gradient accumulation, 4-bit NF4 quantization, bf16 compute dtype.

**Don't waste time tuning:** an exhaustive learning-rate/rank grid search, warmup schedule variants, or optimizer choice beyond the TRL/PEFT defaults — none of these are where your research value comes from at this scale. If you do one sweep, make it a small rank comparison (8 vs. 16 vs. 32) on the joint run only, not across every ablation.

---

## 14. Pragmatics-Aware Fine-Tuning — Decision

| | Approach A: plain SFT | Approach B: pragmatics-aware targets | Approach C: custom auxiliary loss |
|---|---|---|---|
| What it is | Fine-tune on (input, gold label) pairs | Fine-tune on (input, gold label + short reasoning/intended-meaning explanation) | Standard LM loss + a hand-designed extra loss term |
| Advantage | Simplest, fastest to implement | Encodes *why*, not just *what* — teaches the reasoning pattern, still uses standard tooling | Theoretically most targeted intervention |
| Disadvantage | May learn label shortcuts without real reasoning | Slightly more annotation work (need reasoning text, not just labels) | Hard to implement correctly, hard to debug, uncertain payoff over B |
| Difficulty | Low | Low–Medium | High |
| Research value | Low (least differentiated) | Medium–High (defensible "pragmatics-aware" claim, standard technique) | Medium (novel but risky) |

**Decision: Approach B.** It's what your abstract's "pragmatics-aware" language should actually mean — targets that include a brief intended-meaning justification, not just a bare label — using entirely standard SFT tooling (TRL's `SFTTrainer`, no custom loss code). This is scientifically honest, implementable in your timeline, and still differentiates you from "we ran vanilla classification fine-tuning." Approach C stays OPTIONAL/STRETCH only, exactly as flagged in the prior blueprint — don't promise it in your abstract.

---

## 15. The Main Research Experiment — Cross-Category Transfer

**Experiment matrix** (4 training conditions × 3 test categories = 12 cells):

| Trained on ↓ / Tested on → | Sarcasm | Implicature | Indirect |
|---|---:|---:|---:|
| Sarcasm only | (in-category) | transfer? | transfer? |
| Implicature only | transfer? | (in-category) | transfer? |
| Indirect only | transfer? | transfer? | (in-category) |
| Joint (all 3) | in-category | in-category | in-category |

**What each answers:** the diagonal cells (in-category) tell you the ceiling effect of fine-tuning at all. The off-diagonal cells directly answer RQ3 — does training on one phenomenon help another. The joint row is your comparison baseline for "is training on everything better than any single category, including for categories it wasn't specifically trained on."

**Interpreting outcomes:**
- **Positive transfer:** off-diagonal accuracy improves over the *zero-shot baseline* for that test category (not just over random/chance).
- **No transfer:** off-diagonal accuracy stays statistically indistinguishable from baseline.
- **Negative transfer / catastrophic interference:** off-diagonal accuracy is significantly *worse* than baseline — the model overfit to one phenomenon's surface patterns in a way that hurts general pragmatic reasoning.

All three outcomes are reportable, legitimate findings — say this explicitly in your report so the team doesn't feel pressure to only report positive transfer.

**Realistic scope given compute:** if GPU time is genuinely tight, the joint run (MUST DO) plus **one** single-category run (SHOULD DO, pick sarcasm — likely the most-data-rich category) still lets you report a partial transfer matrix (a 2×3 slice) — a valid fallback, not a failure (see Section 26).

---

## 16. Context Ablation

**Design:** for the same test items, run evaluation twice — once with the full `context` field included in the prompt, once with `context` stripped (utterance only) — for both baseline and fine-tuned models.

**What it tells you:** if accuracy barely drops when context is removed, the model (or your benchmark) may be more solvable from surface cues than you think — genuinely useful, if uncomfortable, information about benchmark validity. If accuracy drops sharply with context removed, that's evidence the model is (or, post-fine-tuning, has learned to be) actually using context, not just pattern-matching the utterance.

**Expected result table shape:**

| Model | Context | Sarcasm Acc | Implicature Acc | Indirect Acc |
|---|---|---:|---:|---:|
| Baseline | With context | 0.64 | 0.58 | 0.71 |
| Baseline | Utterance only | 0.55 | 0.52 | 0.68 |
| Fine-tuned | With context | 0.78 | 0.70 | 0.80 |
| Fine-tuned | Utterance only | 0.62 | 0.55 | 0.70 |

A *larger* context-drop for the fine-tuned model than the baseline (as in the illustrative numbers above) would suggest fine-tuning increased genuine context-reliance — an interesting, reportable secondary finding.

---

## 17. Demonstration Application

**Architecture:**
```
User input (context, utterance)
        ↓
Gradio frontend
        ↓
Inference function (loads base model once at startup;
loads LoRA adapter via peft; runs both variants per request)
        ↓
Baseline prediction  |  Fine-tuned prediction  |  Gold answer (if a preset example)
        ↓
Side-by-side comparison table rendered in the Gradio UI
```

**Frontend recommendation: Gradio.** Simplest option that still looks professional, runs directly from your training environment or a Hugging Face Space, and is the standard demo format for HF-model projects — no separate backend framework needed.

**Fields shown per prediction:** pragmatic category, literal meaning, predicted intended meaning, confidence (self-reported or softmax over options), correct/incorrect flag when a gold answer exists.

**Preset examples:** include at least 5–6 curated examples spanning all 3 categories, plus **at least one where the fine-tuned model is still wrong** — this is more credible in front of faculty than an all-wins demo set, and directly supports your limitations discussion.

---

## 18. Proving the Demo Works — 5-Minute Faculty Script

1. **(30s)** One-sentence framing: "We're testing whether LLMs understand what people mean, not just what they say, across sarcasm, implicature, and indirect requests."
2. **(60s)** Show the baseline model failing on a clear example — e.g., a sarcastic sentence the baseline model reads as sincere. Show the gold answer.
3. **(60s)** Same input, fine-tuned model → correct answer. Point out the confidence/label change.
4. **(60s)** Show the transfer-matrix heatmap: "Fine-tuning on sarcasm alone also improved implicature performance by X points — that's evidence pragmatic understanding partially generalizes, not just memorizes one task."
5. **(45s)** Show the context-ablation table: "Removing context hurts the fine-tuned model more than the baseline — meaning it's actually learned to use context, not just pattern-match words."
6. **(30s)** Show one honest failure case in the demo, and name it as a limitation.
7. **(15s)** Close with the one-sentence research contribution: "the cross-category transfer result."

**Prepare in advance:** baseline-vs-fine-tuned results table (Section 11), transfer-matrix heatmap (Section 15), context-ablation table (Section 16), 2–3 failure-analysis example screenshots (Section 12), the live demo tested on a clean environment restart the day before.

---

## 19. Real-World Company Use Cases

**1. Customer support** — *Problem:* sentiment/intent classifiers score sarcastic complaints as neutral/positive. *Pragmatic failure:* literal-polarity reading of sarcasm. *Solution:* a pragmatic-interpretation pass before ticket routing/sentiment scoring. *Integration:* sits as a pre-processing layer in front of an existing intent classifier, re-scoring or flagging ambiguous cases for human review. *Business value:* fewer misrouted/under-prioritized complaints. *Limitation:* needs domain-specific fine-tuning data (support transcripts), not just general pragmatics benchmark data.

**2. Conversational AI / assistants** — *Problem:* literal execution of indirect requests. *Failure:* treats "can you..." as a capability question. *Solution:* an indirect-speech-act classifier before the intent-execution step. *Integration:* pragmatic layer between speech-to-text/NLU and the action-execution engine. *Value:* more natural, less frustrating interactions. *Limitation:* real assistants need much broader intent coverage than 3 phenomena — this is one component, not the whole NLU stack.

**3. Social media analytics** — *Problem:* sarcasm is the classic sentiment-analysis failure mode, especially on Twitter/X-style short text. *Solution:* sarcasm-aware re-scoring layer. *Integration:* post-processing step after standard sentiment models. *Value:* more accurate brand/sentiment tracking. *Limitation:* your model is tuned on curated data, not messy, meme-heavy real social text — would need retraining on in-domain data.

**4. Content moderation** — *Problem:* indirect hostility/mitigated disagreement evades keyword/toxicity filters. *Solution:* the failure taxonomy's categories map onto exactly the ambiguous cases worth flagging for human review, rather than trying to auto-resolve them. *Integration:* a "flag for review" signal layered on top of existing moderation pipelines, not a replacement decision-maker. *Value:* catches indirect harassment patterns. *Limitation:* high stakes for false positives/negatives — would need much more rigorous validation before any autonomous action.

**5. Education** — *Problem:* tutoring systems misread hedged/uncertain student responses as confident (or vice versa). *Solution:* implicature-aware response calibration. *Integration:* a signal into the tutoring system's difficulty/pacing logic. *Value:* better-adapted instruction. *Limitation:* students' hedging patterns vary a lot by age/language background — general benchmark won't transfer directly.

**6. HR/recruitment** — *Problem:* indirect feedback in interview transcripts/references ("she's punctual") gets read at face value. *Solution:* an implicature-aware summarization pass on reference checks. *Integration:* assistive tool for a human reviewer, never an autonomous hiring decision. *Value:* surfaces implied concerns a literal read misses. *Limitation:* legal/fairness risk is high in hiring contexts — this is a strong "assistive only" use case, explicitly not an autonomous-decision one.

**7. Healthcare communication** — *Problem:* patients often communicate concerns indirectly ("I've just been really tired lately") rather than as direct complaints. *Solution:* an indirect-speech-act/implicature flag in intake-chatbot transcripts, surfaced to clinicians. *Integration:* assistive triage signal, not diagnostic. *Value:* fewer missed concerns in asynchronous/chat-based intake. *Limitation:* this is the highest-stakes use case listed — would need extensive clinical validation and is far beyond what a benchmark-scale research prototype could support; mention it as a direction, not a claim.

**8. Enterprise assistants** — *Problem:* internal chat assistants (meeting-notes bots, ticketing bots) miss indirect requests/implied priorities in employee messages. *Solution:* same indirect-speech-act layer as use case 2, applied to workplace text. *Integration:* pre-processing layer feeding the assistant's task-creation logic. *Value:* fewer missed action items. *Limitation:* enterprise text has its own jargon/conventions your general benchmark won't fully capture.

**Deployment reality check:** in every case above, a company would use a model like yours as **one component in a larger pipeline** (a pragmatic-interpretation layer feeding into existing intent/classification/business-logic systems), never as a standalone deployed decision-maker — say this explicitly in your report, it's the correct and expected framing for a research prototype.

---

## 20. Production Architecture — Prototype vs. Enterprise

**Student prototype:**
```
User → Gradio → Fine-tuned LLM (LoRA adapter) → Prediction
```

**Enterprise system (illustrative):**
```
User message
   → preprocessing (normalization, PII handling)
   → pragmatic analysis layer  ←── this is where your model/approach fits
   → intent detection
   → safety checks
   → business logic / routing
   → response generation
   → monitoring & logging
```

Your contribution occupies exactly one box in the enterprise diagram — the **pragmatic analysis layer** — feeding a *signal* (category + intended meaning + confidence) into whatever intent/business-logic system already exists, rather than replacing it. Be precise about this scope in your report; it's both more honest and, frankly, more credible to a technical reviewer than implying an end-to-end deployable system.

---

## 21. Project Requirements

**Functional:** benchmark exists and is queryable; harness evaluates any HF-compatible or API model on it; fine-tuning pipeline produces a loadable adapter; demo runs a live comparison.

**Non-functional:** reproducible (fixed seeds/configs), runs on free/low-cost GPU tier, evaluation harness runtime reasonable (< a few hours per full model pass).

**ML requirements:** QLoRA fine-tuning support, deterministic-as-possible evaluation, statistical significance testing.

**Dataset requirements:** unified schema, licensed sources, ≥100 hand-verified test items, κ-scored annotation.

**Hardware requirements:** single 16GB+ VRAM GPU (Colab/Kaggle-tier acceptable).

**Software requirements:** per Section 9 stack.

**Research requirements:** finalized RQs, failure taxonomy applied, transfer matrix + context ablation + significance testing completed.

**Documentation requirements:** per Section 2's documentation deliverables.

**Demonstration requirements:** working Gradio app, tested on a clean environment restart, honest example set including a failure case.

---

## 22. Feature Scope

**MVP:** unified benchmark, baseline eval (3 models), joint QLoRA fine-tuned model, post-FT eval, basic accuracy/F1 comparison, working demo.

**Research features:** McNemar's significance testing, systematic failure taxonomy, at least a partial transfer matrix (joint + ≥1 single-category run), context ablation.

**Advanced/optional:** full 4-condition transfer matrix (all 3 single-category runs), LoRA-rank sweep, second base model fine-tuned, custom auxiliary loss (Section 14, Approach C), public HF Space deployment.

---

## 23. Repository Structure

```text
pragmatic-llm/
├── data/
│   ├── raw/              # created Phase 1 — untouched downloads
│   ├── processed/        # created Phase 2 — normalized schema
│   └── benchmark/        # created Phase 4 — frozen splits
├── src/
│   ├── preprocessing/    # Student A — Phases 1-4
│   ├── evaluation/       # Student B — Phases 5, 8
│   ├── training/         # Student C — Phase 7
│   └── analysis/         # Student B — Phases 6, 9
├── configs/               # created alongside first use in each phase (prompts, LoRA configs)
├── notebooks/             # exploratory only, all owners, ongoing
├── experiments/           # created Phase 7 — one subfolder per fine-tuning run
├── app/                   # Student C — Phase 10
├── tests/                 # schema validation (Phase 2), harness sanity checks (Phase 5)
├── docs/                  # created Phase 0 onward, all owners
├── requirements.txt        # created Phase 0, updated as dependencies are added
└── README.md               # created Phase 0 (skeleton), completed Phase 11
```

**Files to create first (Week 1):** `README.md` (skeleton), `requirements.txt`, `docs/research_questions.md`, `src/preprocessing/download.py`.

---

## 24. Team Division (3 students)

- **Student A — Dataset & Benchmark:** Phases 1–4 (collection, cleaning, annotation, freeze). Owns `data/`, `src/preprocessing/`.
- **Student B — Evaluation & Analysis:** Phases 5, 6, 8, 9 (baseline eval, failure analysis, post-FT eval, transfer/ablation/stats). Owns `src/evaluation/`, `src/analysis/`.
- **Student C — Fine-Tuning & Demo:** Phase 7, 10 (QLoRA training, Gradio app). Owns `src/training/`, `app/`.

All three co-own Phase 0 (lit review) and Phase 11 (writing, each covering their own phases). This keeps each student's work independently demoable (Student A can show a validated benchmark even before fine-tuning exists; Student B's harness works on any model, so they're not blocked waiting on Student C) while still integrating into one pipeline through the shared `data/benchmark/` and `results/` conventions.

**For 2 students:** merge B+C (both are technically heavy but sequential — B's harness work directly feeds C's evaluation-after-training work) and have A remain dataset-focused, since annotation/QC genuinely benefits from not being time-split with ML engineering.

---

## 25. Timeline (One Semester, ~16–18 Weeks)

| Weeks | Focus | Notes |
|---|---|---|
| 1–2 | Phase 0 (lit review, gap, RQs) | Critical deadline: RQs signed off by end of week 2 |
| 2–4 | Phase 1 (dataset collection) | Start indirect-speech-act curation immediately — it's the bottleneck |
| 4–5 | Phase 2 (cleaning/normalization) | Overlaps with tail of Phase 1 |
| 5–7 | Phase 3 (annotation/QC) | Parallel across all 3 students for the double-annotation sample |
| 7 | Phase 4 (benchmark freeze) | Hard gate — nothing after this touches the splits without a version bump |
| 8–9 | Phase 5 (baseline eval) | Student B; Student C starts training-data formatting in parallel |
| 9–10 | Phase 6 (pre-FT failure analysis) | Student B |
| 9–11 | Phase 7 (fine-tuning, joint run) | Student C; buffer for OOM/debugging built in |
| 11–12 | Phase 7 (per-category runs, if time allows) | SHOULD DO — cut first if behind schedule |
| 12–13 | Phase 8 (post-FT eval) | Student B/C |
| 13–14 | Phase 9 (transfer matrix, ablation, stats) | The core research payoff — protect this time, don't let it get squeezed |
| 14–15 | Phase 10 (demo) | Student C; test on clean environment by week 15 |
| 15–17 | Phase 11 (report/paper) | Start drafting methodology earlier (week 8+), don't wait |
| 17–18 | Buffer + review prep | Rehearse the 5-minute demo script (Section 18) |

**Parallel work:** dataset (A) and lit review (all) in weeks 1–4; evaluation harness build (B) and training-data prep (C) in weeks 7–9; report drafting (all) starting week 8 alongside experiments, not after.

**Buffer:** weeks 17–18 are explicit buffer — if Phase 7's per-category runs slipped, this is where they'd need to be cut, not the demo or the write-up.

---

## 26. Risk Table

| Risk | Probability | Impact | Solution |
|---|---|---|---|
| GPU limitations / Colab disconnects | High | Medium | QLoRA + checkpointing + resume logic; use Kaggle's weekly GPU quota as backup |
| Dataset licensing/availability issues | Medium | Medium | Verify at download time; Plan B sources per Section 5 |
| Indirect-speech-act sourcing takes too long | High | Medium | Start in week 1–2, not after sarcasm/implicature are done |
| Training failure (OOM, divergence) | Medium | Medium | Small batch + grad accumulation; monitor val loss; keep last-good checkpoint |
| Overfitting on small fine-tuning set | Medium | Medium | Held-out val set, early stopping, modest LoRA rank |
| Poor/no improvement from fine-tuning | Medium | High if unplanned for | **See fallback below — this has a valid scientific response, it isn't a project failure** |
| Negative transfer in one category | Medium | Low (it's a valid finding) | Report it as a finding, discuss in failure analysis |
| Benchmark contamination | Low–Medium | Medium | Weight claims toward hand-curated test subset |
| API cost overrun | Low | Low | Fixed call cap, caching, no reruns |
| Demo latency at review | Medium | Low | Pre-load models at app startup, not per-request; test 4-bit inference speed ahead of time |
| Statistical significance not reached | Medium–High (small test sets) | Low if reported honestly | Report effect size + "not significant at this sample size" rather than omitting or overselling |

**What to do if fine-tuning does NOT improve the model — the fallback plan:** this is a legitimate, presentable outcome, not a dead end. If joint fine-tuning shows no significant gain: (1) report it honestly with the McNemar's p-value, (2) pivot the narrative toward whichever secondary finding *did* show something — e.g., a context-ablation effect, or a specific failure-taxonomy pattern that explains *why* fine-tuning didn't move the needle (this is often more interesting than a plain accuracy bump), (3) check whether the null result is uniform or category-specific — a "helped sarcasm, didn't help implicature" pattern is itself informative about the RQ2 category-difference question, (4) as a last resort, a well-documented negative result with a clear methodological explanation is a defensible B.Tech research outcome — plan the report structure so this doesn't require last-minute rewriting if it happens.

---

## 27. Success Criteria

**Minimum successful project:** benchmark exists and is schema-valid; baseline evaluation runs and produces results on ≥2 models; fine-tuned model exists and loads; demo runs end-to-end; report describes the pipeline honestly regardless of numeric outcome.

**Good project:** the above, plus a measurable (even if not statistically significant) accuracy change from fine-tuning; failure taxonomy applied and reported; at least a partial transfer matrix (joint + 1 category).

**Excellent research project:** the above, plus a statistically tested transfer-matrix finding (positive, null, or negative — any of these, well-supported, counts), a context-ablation result, reproducible configs/seeds throughout, and a research paper draft whose claims are all traceable to specific result files.

No accuracy targets are specified here on purpose — a defensible, honestly-reported small effect is a better outcome than an inflated claim chasing an arbitrary number.

---

## 28. What We Should NOT Build

- **Custom LLM from scratch** — no compute for it, and it isn't what your RQs need.
- **A large custom dataset built entirely from scratch** — existing sarcasm/implicature datasets are appropriate and well-annotated; rebuilding them wastes your annotation budget, which is scarce and better spent on indirect speech acts.
- **A database** — JSONL files + pandas cover this project's scale entirely.
- **Microservices / separate backend framework** — Gradio's built-in server is sufficient; Flask/FastAPI adds complexity with no benefit here.
- **A mobile app** — out of scope for a research demo; adds platform-specific work with zero research value.
- **A complicated custom frontend (React, etc.)** — Gradio is faster to build and just as demo-ready.
- **A custom loss function as a plan dependency** — real engineering risk for uncertain payoff (Section 14); keep it OPTIONAL only.
- **More than ~3 evaluation models + 1 fine-tuning target** — each additional model is more eval time without proportionally more research signal.
- **A large hyperparameter grid search** — spend that time budget on the transfer study instead (Section 25's protected time block).
- **Unnecessary cloud infrastructure** — Colab/Kaggle is sufficient; don't set up cloud VMs/managed training services for this scale.

---

## 29. Final Recommended Architecture (One Version, Build This)

- **Dataset:** combine MUStARD + iSarcasm (sarcasm) and Circa (implicature); hand-curate indirect speech acts (start earliest); unified schema per Section 6; ~600–1,300 items total; κ-scored annotation with sincere/literal controls.
- **Benchmark:** train/val/test split at source-dataset level; ~50–100-item hand-verified, contamination-checked test subset weighted most heavily in headline claims.
- **Models:** Llama-3.1-8B-Instruct + Mistral-7B-Instruct + one API reference model for baseline; Llama-3.1-8B-Instruct as the sole QLoRA fine-tuning target.
- **Fine-tuning:** QLoRA, Approach B (pragmatics-aware instruction targets with brief reasoning, standard SFT loss, no custom loss function). Joint run is MUST DO; per-category runs are SHOULD DO, cut first if time-constrained.
- **Experiments:** transfer matrix (full if time allows, partial 2×3 slice as fallback), context ablation, zero/few-shot/fine-tuned comparison, McNemar's significance testing throughout.
- **Metrics:** accuracy, macro-F1, confusion matrix, McNemar's p-values as primary; LLM-as-judge (human-validated) for generation subset as secondary.
- **Demo:** Gradio, baseline-vs-fine-tuned side-by-side, honest example set including a failure case.
- **Tech stack:** per Section 9, nothing added beyond it.
- **Repository:** per Section 23.
- **Team:** per Section 24 (3-way split, Student A/B/C).
- **Timeline:** per Section 25, with weeks 13–14 (transfer/ablation/stats) protected as the non-negotiable research core.

---

## 30. Final Master Checklist

**Phase 0**
- [ ] Re-read all 5 cited papers, extract exact evaluated phenomena/datasets/metrics
- [ ] Write gap statement, get whole-team sign-off
- [ ] Finalize RQ1–RQ5 + H1–H3 in `docs/research_questions.md`

**Phase 1**
- [ ] Download/verify MUStARD, iSarcasm, Circa (check current license/availability)
- [ ] Begin indirect-speech-act curation immediately (bottleneck category)
- [ ] Log all licenses in `data/raw/LICENSES.md`

**Phase 2**
- [ ] Write and run `src/preprocessing/normalize.py` for each source
- [ ] Validate every record against the unified schema
- [ ] Deduplicate

**Phase 3**
- [ ] Write `docs/annotation_guidelines.md`
- [ ] Run calibration round, refine guideline
- [ ] Double-annotate 10–15% sample, compute κ (target ≥0.6)
- [ ] Add sincere/literal control items
- [ ] Tag `difficulty` / `requires_context`; flag ambiguous items

**Phase 4**
- [ ] Split train/val/test at source-dataset level
- [ ] Run contamination/leakage check
- [ ] Freeze splits, write `docs/dataset_card.md`

**Phase 5**
- [ ] Implement `src/evaluation/harness.py` + 2 prompt-template variants
- [ ] Run zero-shot + few-shot across all 3 baseline models
- [ ] Store all results in the schema from Section H of the original blueprint

**Phase 6**
- [ ] Auto-tag context-under-use and lexical-shortcut failures
- [ ] Manually tag ≥15 errors/category/model
- [ ] Produce failure-distribution charts

**Phase 7**
- [ ] Format training data (Approach B: label + brief reasoning)
- [ ] Configure and run joint QLoRA fine-tuning (MUST)
- [ ] Run single-category fine-tunes if time allows (SHOULD)
- [ ] Save all adapters + configs under `experiments/`

**Phase 8**
- [ ] Rerun exact Phase 5 harness on each fine-tuned adapter
- [ ] Produce baseline-vs-fine-tuned comparison table

**Phase 9**
- [ ] Build transfer matrix (full or partial per Section 15)
- [ ] Run context-ablation comparison
- [ ] Run McNemar's test on all key pairs, record p-values
- [ ] Produce transfer-matrix heatmap

**Phase 10**
- [ ] Build Gradio demo app
- [ ] Curate example set (incl. ≥1 honest failure case)
- [ ] Test on a clean environment restart

**Phase 11**
- [ ] Draft review paper (related work + gap)
- [ ] Draft research paper (method, results, transfer-matrix discussion, limitations)
- [ ] Draft final report
- [ ] Team review pass before submission
- [ ] Rehearse the 5-minute demo script (Section 18)

---
*This is the execution-ready version of the project plan. It supersedes the earlier blueprint's phase/experiment scoping where the two differ (Section 0). Ready to be converted into a repo-ready `MASTER_PROJECT_PLAN.md` on request.*
