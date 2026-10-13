# Measured results

These values describe the committed TF-IDF logistic-regression CPU reference, not Llama or QLoRA. Source artifacts are [metrics](../results/local/metrics.json) and [predictions](../results/local/predictions.jsonl).

| Category | Test examples | Context accuracy | Context macro-F1 | No-context accuracy | No-context macro-F1 |
|---|---:|---:|---:|---:|---:|
| Sarcasm | 104 | 62.50% | 0.6091 | 59.62% | 0.5924 |
| Implicature | 4,726 | 57.26% | 0.5575 | 61.24% | 0.5897 |
| Indirect speech acts | 22 | 50.00% | 0.4078 | 40.91% | 0.3300 |

The two input conditions use separately fitted CPU classifiers. Context helps on two categories in these measurements and reduces the implicature score; these observations are not evidence that LLM fine-tuning improves pragmatic competence. The uneven task sizes make pooled accuracy dominated by implicature. Report task-level scores and support.

The graphs page recalculates scores from saved predictions and displays coverage. Full Llama baseline, joint/category adapter results, cross-category transfer and independent interpretation scoring remain unmeasured in the available research evidence. Missing scores must not be filled with estimates or synthetic test values.
