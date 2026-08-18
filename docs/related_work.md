# Related Work
## Evaluating and Enhancing Pragmatic Competence in LLMs

---

> **Status:** Skeleton — to be completed during Phase 0 (Literature Review).
> Each section below should be expanded with specific paper citations, datasets, metrics, and findings.

---

## 1. Pragmatics in NLP — Background

### 1.1 Semantics vs. Pragmatics
- Gricean maxims (quality, quantity, relation, manner) and their relevance to NLU
- Speech act theory (Austin, Searle) — locutionary, illocutionary, perlocutionary acts
- Computational pragmatics as a distinct NLP challenge

### 1.2 Why LLMs Struggle with Pragmatics
- Training objective (next-token prediction) favors semantic coherence over pragmatic interpretation
- Lack of explicit speaker-intent modeling
- Context window limitations vs. conversational context requirements

---

## 2. Sarcasm and Irony Detection

### 2.1 Datasets
- **iSarcasm** — [citation needed] — size, labels, context availability
- **MUStARD** — [citation needed] — multimodal, dialogue context
- **SemEval shared tasks** — [citation needed] — various years
- **Sarcasm Corpus V2** — [citation needed]

### 2.2 Methods and Findings
- Traditional ML approaches (feature engineering, sentiment lexicons)
- Transformer-based approaches
- LLM zero-shot/few-shot performance
- Key limitations and error patterns

---

## 3. Conversational Implicature

### 3.1 Datasets
- **Circa** — [citation needed] — indirect answers to polar questions, size, label scheme
- **Goldilocks paper datasets** — [citation needed] — implicature-specific fine-tuning study
- **GRICE-style curated sets** — [citation needed]

### 3.2 Methods and Findings
- Human-vs-LLM comparisons on implicature resolution
- Fine-tuning strategies and their effectiveness
- Scale effects on implicature comprehension
- Relationship to instruction-tuning

---

## 4. Indirect Speech Acts

### 4.1 Datasets
- Limited ready-made resources — the weakest category for available benchmarks
- Politeness-strategy corpora (indirect-request subsets)
- Template-derived variants (direct vs. indirect phrasing)

### 4.2 Methods and Findings
- Speech act classification approaches
- Mitigated disagreement pragmatics — [citation needed]
- Non-verbal pragmatic inference — [citation needed]

---

## 5. Cross-Phenomenon and Unified Evaluation

### 5.1 Existing Unified Approaches
- [Identify any papers attempting multi-phenomenon pragmatic evaluation]
- Current state: largely siloed per phenomenon

### 5.2 Transfer Learning for Pragmatics
- [Identify papers studying cross-task transfer in pragmatic/social NLU tasks]
- Current state: under-explored — no cited paper directly answers "does implicature fine-tuning help sarcasm detection?"

---

## 6. Parameter-Efficient Fine-Tuning for NLU Tasks

### 6.1 LoRA and QLoRA
- [LoRA paper citation]
- [QLoRA paper citation]
- Effectiveness on classification and instruction-following tasks

### 6.2 Fine-Tuning for Pragmatic / Social Understanding
- [Goldilocks paper — fine-tuning strategies for implicature]
- Chain-of-thought fine-tuning as "pragmatics-aware" supervision

---

## 7. Identified Gap

**No existing work provides:**
1. A unified, same-format, same-metric evaluation across sarcasm + implicature + indirect speech acts on the same models.
2. A cross-category transfer study answering whether pragmatic fine-tuning generalizes across phenomena.
3. A structured failure taxonomy applied consistently across all three categories.

**This project fills gap #2 as its primary contribution**, with #1 as infrastructure and #3 as supporting analysis.

---

## References

> To be populated during Phase 0. Use a consistent citation format (e.g., APA or ACL anthology style).

1. [Paper 1 — TBD]
2. [Paper 2 — TBD]
3. [Paper 3 — TBD]
4. [Paper 4 — TBD]
5. [Paper 5 — TBD]
