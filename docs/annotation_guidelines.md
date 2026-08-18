# Annotation Guidelines
## Pragmatic Competence Benchmark

---

## Purpose

These guidelines ensure consistent, reproducible labeling across all three pragmatic phenomena in our benchmark. Every annotator must read this document **before** labeling any items.

---

## General Rules

1. **Read the full context first**, then the utterance, then decide.
2. **Label based on what a reasonable adult English speaker would understand**, not on what is logically possible.
3. **When genuinely uncertain**, mark `difficulty: hard` and add a note in `annotator_notes` — do NOT force a label you aren't confident in.
4. **Do not look at other annotators' labels** before completing your own pass.
5. **Flag items that feel ambiguous** even after careful reading — these are valuable data points for the "ambiguous" category, not annotation failures.

---

## Category 1: Sarcasm / Irony

### Labels
- `sarcastic` — the speaker's intended meaning is the **opposite** (or substantially different from) the literal meaning, conveyed through irony or sarcasm.
- `sincere` — the speaker means what the words literally say.

### Decision Procedure
1. Read the context: does it set up a situation that **contradicts** the literal meaning of the utterance?
2. Would a reasonable listener in this context understand the speaker as meaning the opposite?
3. If yes → `sarcastic`. If the literal reading is plausible and intended → `sincere`.

### Worked Examples

**Example 1 — Sarcastic**
- Context: *"A student fails every question on the exam."*
- Utterance: *"Wow, you absolutely nailed that."*
- Label: `sarcastic`
- Intended meaning: *"The student did terribly."*
- Reasoning: Context (total failure) contradicts the literal praise. Any reasonable listener would recognize the irony.

**Example 2 — Sincere**
- Context: *"A student gets 98% on the exam."*
- Utterance: *"Wow, you absolutely nailed that."*
- Label: `sincere`
- Intended meaning: *"The student did excellently."*
- Reasoning: Context (near-perfect score) is consistent with the literal praise. Same words as Example 1, but the context makes it genuine.

**Example 3 — Edge case**
- Context: *"A student gets 72% on a difficult exam."*
- Utterance: *"Well, that's... impressive."*
- Label: Could be either — if ambiguous to you, mark `difficulty: hard` and add a note about what makes it unclear.

### Common Pitfalls
- **Don't rely on trigger words alone.** "Great," "wonderful," "brilliant" are NOT automatically sarcastic — context determines this.
- **Deadpan sarcasm exists.** The utterance doesn't need exclamation marks or exaggeration to be sarcastic.

---

## Category 2: Conversational Implicature

### Labels
- `yes` — B's reply pragmatically implies a **positive/affirmative** answer to A's question.
- `no` — B's reply pragmatically implies a **negative** answer to A's question.
- `uncertain` — B's reply is genuinely ambiguous or non-committal; a reasonable listener could not confidently assign yes or no.

### Decision Procedure
1. Read A's question.
2. Read B's reply — does it directly answer the question? (If yes, this may not be an implicature item.)
3. If B's reply is indirect: what would a reasonable listener infer B *means* about A's question?
4. Apply the Gricean relevance test: *why would B say this, if not to imply X?*

### Worked Examples

**Example 1 — Implicature: "no"**
- A: *"Did you like the movie?"*
- B: *"The popcorn was great."*
- Label: `no`
- Intended meaning: *"B didn't like the movie — deflecting to the popcorn implies the movie itself wasn't worth commenting on positively."*
- Reasoning: B's reply is relevant only if it's avoiding a direct negative answer. Gricean relevance: if B liked the movie, B would say so.

**Example 2 — Implicature: "yes"**
- A: *"Would you recommend the restaurant?"*
- B: *"I've already been back three times."*
- Label: `yes`
- Intended meaning: *"B recommends it — returning multiple times implies strong approval."*

**Example 3 — Uncertain**
- A: *"Did you enjoy the concert?"*
- B: *"The seats were comfortable."*
- Label: `uncertain`
- Reasoning: Could be a polite deflection (implying the concert itself wasn't great) or could genuinely mean the seats were the highlight. Annotators may disagree — that's fine, mark `difficulty: hard`.

### Common Pitfalls
- **Don't over-infer.** If B's reply is genuinely ambiguous, label `uncertain` rather than guessing.
- **Cultural assumptions:** some implicatures depend on cultural norms. Note when this is the case.

---

## Category 3: Indirect Speech Acts

### Labels
- `statement` — the utterance functions as a factual statement (literal interpretation).
- `request` — the utterance functions as a request or command.
- `question` — the utterance functions as a genuine information-seeking question.
- `suggestion` — the utterance functions as a suggestion or recommendation.
- `complaint` — the utterance functions as a complaint or expression of dissatisfaction.

### Decision Procedure
1. What is the **grammatical form** of the utterance? (Question? Statement? Exclamation?)
2. In this context, what **action or response** would a reasonable listener take?
3. If the expected response doesn't match the grammatical form → it's an indirect speech act. Label by **function**, not form.

### Worked Examples

**Example 1 — Indirect Request**
- Context: *"Two coworkers in an office."*
- Utterance: *"It's really cold in here."*
- Label: `request`
- Intended meaning: *"Close the window or turn up the heat."*
- Reasoning: Grammatically a statement, but in context, a reasonable listener would act on it as a request.

**Example 2 — Genuine Statement**
- Context: *"A weather reporter on TV."*
- Utterance: *"It's really cold out there today."*
- Label: `statement`
- Intended meaning: *"Informing the audience about the weather."*
- Reasoning: Same structure as Example 1, but the speaker role and context make it a genuine informational statement.

**Example 3 — Indirect Suggestion**
- Context: *"Two friends discussing weekend plans."*
- Utterance: *"I hear that new Italian place is really good."*
- Label: `suggestion`
- Intended meaning: *"We should go to that Italian restaurant."*

### Common Pitfalls
- **"Can you...?" and "Could you...?" are almost always indirect requests**, not genuine ability questions. But verify with context — in rare cases they ARE genuine questions (e.g., doctor asking about physical ability).
- **Multiple functions:** some utterances serve dual purposes (e.g., a complaint that's also a request). Pick the **primary** function; note the secondary one in `annotator_notes`.

---

## Quality Control Process

1. **Calibration round:** All annotators label the same 10 items independently, then compare and resolve disagreements together. Refine these guidelines based on calibration findings.
2. **Double annotation:** 10–15% of all items are labeled by two annotators independently.
3. **Cohen's κ:** Computed on the double-annotated sample, per category. Target: κ ≥ 0.6.
4. **Tiebreaker:** For disagreements, a third annotator decides. If still unresolved → `difficulty: hard`, `label: ambiguous`.
5. **Deduplication:** Run before splitting — near-identical utterances (fuzzy match) must not appear in both train and test.

---

## Metadata Fields

| Field | How to fill it |
|---|---|
| `difficulty` | `easy` = resolvable from obvious contextual cues; `medium` = requires inference but is clear; `hard` = requires world knowledge or is near-ambiguous |
| `requires_context` | `true` if removing the context would make the correct label unrecoverable; `false` if the utterance alone is sufficient |
| `annotator_notes` | Free text. Use for: why you chose a label on a borderline item, cultural assumptions required, secondary speech-act function, etc. |
