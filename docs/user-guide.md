# User guide

## Analyze an utterance

Open the home page, enter context and an utterance, choose the pragmatic category, and select Analyze. The category is supplied by the user; the model predicts a label within that category. Presets provide illustrative gold labels and reference meanings.

In `--local` mode the comparison is a training-majority baseline against TF-IDF logistic regression. Reference meanings displayed for presets are not generated explanations. In model mode, genuine adapter comparison requires both the base model and an existing adapter. `--no-model` is an explicitly labeled scripted simulation.

## View graphs and matrices

Select **View graphs and result matrices** to open `/graphs`. Select a category to inspect its scores and confusion matrices. Tabs contain test results, per-label precision/recall/F1, confusion matrices, live training loss, dataset counts and the LLM transfer matrix. Missing transfer cells do not mean zero accuracy.

Saved research files refresh every five seconds. Refresh research evidence performs a manual refresh. The live-input graph counts the last 100 predicted labels stored in this browser; it does not update benchmark scores or model weights. Clear prediction history removes that history. Input text is not stored by this history feature.

The requirements table has been removed from the graph interface. The written [completion checklist](completion-status.md) remains available for research planning.

## Human interpretation review

Open `/review`, read the displayed rubric, enter a pseudonymous reviewer ID and select Next item. Read the context, utterance and generated interpretation. Rate meaning, contextual use and grounding from 0 to 2, explain your ratings, confirm independent review and save. Select Next item to continue.

The review view hides model identity and other reviewers' scores. It requires real outputs in `results/interpretations`, generated using the matching benchmark. An empty queue means there are no eligible samples; the app does not generate substitute ratings. Use at least two real independent reviewers. IDs alone do not prove independence. Disagreements need a documented adjudication step outside the rating form.

Refresh review coverage to see coverage and pairwise agreement. Undefined agreement statistics remain undefined rather than being shown as perfect agreement.
