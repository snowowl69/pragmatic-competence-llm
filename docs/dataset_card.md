# Dataset Card — Pragmatic Competence Benchmark

> **Status:** Skeleton — to be completed after Phase 4 (Benchmark Freeze).

---

## Dataset Summary

A unified benchmark for evaluating LLM pragmatic competence across three phenomena: sarcasm/irony, conversational implicature, and indirect speech acts. Combines existing public datasets with hand-curated examples in a single schema.

## Supported Tasks

- **Pragmatic category classification** (sarcasm: sincere/sarcastic; implicature: yes/no/uncertain; indirect speech acts: statement/request/question/suggestion/complaint)
- **Intended meaning generation** (secondary, qualitative task)

## Languages

English only.

## Dataset Structure

### Schema
See `docs/annotation_guidelines.md` and `src/preprocessing/schema.py` for the full JSON schema.

### Splits

| Split | Purpose | Size | Notes |
|---|---|---|---|
| `train` | Fine-tuning | TBD | Drawn from combined sources |
| `val` | Hyperparameter tuning, early stopping | TBD | Disjoint at source-dataset level |
| `test` | Final evaluation (never used for training decisions) | TBD | Includes hand-verified, contamination-checked subset |

### Per-Category Distribution

| Category | Train | Val | Test | Total |
|---|---|---|---|---|
| Sarcasm | TBD | TBD | TBD | TBD |
| Implicature | TBD | TBD | TBD | TBD |
| Indirect Speech Acts | TBD | TBD | TBD | TBD |
| **Total** | TBD | TBD | TBD | TBD |

## Source Datasets

| Source | Phenomena | License | Notes |
|---|---|---|---|
| MUStARD | Sarcasm | TBD (verify) | Dialogue context available |
| iSarcasm | Sarcasm | TBD (verify) | Tweet-based |
| Circa | Implicature | TBD (verify) | Indirect answers to polar questions |
| Hand-curated | Indirect speech acts | CC-BY (our own) | Template-derived + manually written |
| Control items | All categories | CC-BY (our own) | Sincere/literal controls for shortcut detection |

## Annotation Process

- Guidelines: `docs/annotation_guidelines.md`
- Double-annotated sample: 10–15% of total
- Inter-annotator agreement: Cohen's κ = TBD (target ≥ 0.6)
- Ambiguous items flagged and excluded from strict accuracy scoring

## Known Limitations

- English-only, predominantly Western cultural context
- Small scale (~600–1,300 items) — not representative of full pragmatic diversity
- Indirect speech act category has the least established source data
- Public-dataset test items may have contamination risk from model pretraining — headline claims weighted toward hand-curated test subset

## Citation

TBD — to be added upon project completion.
