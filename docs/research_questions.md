# Research Questions & Hypotheses
## Evaluating and Enhancing Pragmatic Competence in LLMs

---

## Research Questions

### RQ1 — Baseline Pragmatic Performance
**How accurately, and with what error patterns, do current open- and closed-source LLMs interpret sarcasm, conversational implicature, and indirect speech acts, under a common evaluation protocol?**

- Scope: Zero-shot and few-shot evaluation across all three phenomena using the same prompt templates, metrics, and test set.
- Why it matters: Establishes the "before" picture and quantifies the pragmatic competence gap across model families.

### RQ2 — Cross-Category Performance Differences
**Are there systematic, statistically significant performance differences across the three pragmatic categories, and do they correlate with model scale or training regime (base vs. instruction-tuned)?**

- Scope: Compare per-category accuracy/F1 across models; test whether category ordering (e.g., sarcasm < implicature < indirect) is consistent.
- Why it matters: Reveals whether some pragmatic skills are harder for LLMs categorically, not just anecdotally.

### RQ3 — Cross-Category Transfer (Core Contribution)
**Does parameter-efficient fine-tuning (LoRA) on pragmatic examples from *one* category improve performance on the *other* categories (positive transfer), or does it only help the trained category (or hurt others — catastrophic interference)?**

- Scope: The transfer matrix — train on sarcasm-only, test on all three; repeat for implicature and indirect; compare against joint training.
- Why it matters: This is the primary empirical contribution. No cited prior work directly answers "does implicature fine-tuning help sarcasm detection?" A positive-transfer finding would suggest a shared underlying "look past the literal" skill.

### RQ4 — Context Reliance
**How much does explicit context (vs. utterance-only) contribute to correct pragmatic interpretation, and does this contribution change after fine-tuning?**

- Scope: Context-present vs. context-removed evaluation, before and after fine-tuning.
- Why it matters: Distinguishes genuine context-dependent reasoning from surface-pattern matching. If fine-tuning increases context reliance, the model has learned something deeper than cue words.

### RQ5 — Failure Mode Reduction (Stretch)
**What error types (from the failure taxonomy) are most reduced by fine-tuning, and which persist regardless of adaptation?**

- Scope: Apply the 7-category failure taxonomy to baseline and post-FT errors, compare distributions.
- Why it matters: Provides actionable insight into what fine-tuning actually fixes vs. what requires different interventions.

---

## Hypotheses

### H1 — Category Ordering
Instruction-tuned models will outperform base models of similar size on all three categories, but the *gap between categories* will persist.

**Expected ordering:** indirect speech acts (easiest) > sarcasm > implicature (hardest).

**Rationale:** Indirect speech acts often have syntactic/politeness markers recoverable from the utterance alone. Implicature requires the most context-dependent inference and has the best-documented literal-reading bias in the literature.

### H2 — Transfer and Joint Training
Fine-tuning on all three categories jointly will generalize better than single-category fine-tuning, but even single-category fine-tuning will show **partial positive transfer**, because the phenomena share an underlying "look past the literal string" skill.

**Rationale:** All three phenomena require the model to suppress a literal reading in favor of a context-dependent one — the core cognitive operation is shared, even if surface manifestations differ.

**Null hypothesis:** Single-category fine-tuning shows zero or negative transfer to other categories, suggesting the phenomena are truly independent skills at the representational level.

### H3 — Context Sensitivity
Context removal will hurt sarcasm and implicature performance **more** than indirect speech acts.

**Rationale:** Indirect requests are often recoverable from the utterance's syntax and politeness markers alone (e.g., "Could you...?" is structurally recognizable as an indirect request). Sarcasm requires knowing that the context contradicts the literal tone. Implicature requires knowing what a relevant answer *would* look like to infer the gap.

---

## Gap Statement

**Already well-covered in prior work:**
- Implicature evaluation on individual benchmarks
- Sarcasm/irony detection as a standalone classification task
- Fine-tuning strategy studies for implicature specifically (Goldilocks paper)
- Human-vs-LLM comparisons on implicature
- Mitigated-disagreement pragmatics

**What is thin / under-explored:**
1. **Unified, same-format evaluation** across sarcasm + implicature + indirect speech acts on the *same* models with the *same* metrics, enabling apples-to-apples cross-category comparisons.
2. **Cross-category transfer/generalization** — whether adaptation on one pragmatic phenomenon transfers to the others. This is the most under-explored angle and the core contribution.
3. **Structured, reusable failure taxonomy** applied consistently across all three phenomena.

**Honest assessment:** Simply "combining three phenomena into one framework" is engineering unification, not a research contribution on its own. The cross-category **transfer/generalization study** (RQ3) is what turns this into a real empirical contribution.
