# Research monitoring

Launch `python app/demo.py --local --port 7860 --open-browser` from the repository root.
Select **View graphs and result matrices** to navigate to `/graphs`. The research page refreshes every five seconds and offers manual refresh and task selection. `/review` provides blinded human interpretation scoring. See [completion status](completion-status.md) for the precise distinction between implemented tools and completed experiments.

- **Test results** recomputes accuracy and macro-F1 from saved predictions and shows evaluated/expected coverage. Files, models, prompts, context conditions and run indices stay separate. Partial runs are provisional.
- **Confusion matrices** show actual saved gold/predicted labels. Failed parsing remains an incorrect prediction in the scorer.
- **Dataset** shows the saved train, validation and test category counts.
- **LLM transfer** fills cells only when a condition has predictions for the complete category test set. Missing cells stay missing. Compare models only under equivalent prompts and settings.
- **Live training** reads `experiments/<condition>/live_status.json` and `training_metrics.json`. The QLoRA trainer writes these atomically at optimizer steps and logging events. A stale update is explicitly flagged; it does not imply that a process is still alive.
- **Live inputs** count predictions from the last 100 analyses in browser-local storage, surviving navigation between pages. Clear history on the graph page; server restarts reset it. Input text is not persisted by the monitor. This graph is not test accuracy and does not update model weights.
- **Research status** is documented in the [completion checklist](completion-status.md). The requirements table is no longer displayed on the graph page. File presence alone never certifies the entire research project.

Training on Colab runs on a different machine. Run the dashboard there or copy/sync the `experiments` and `results` folders to this checkout to see those logs here. Local polling cannot observe an unconnected remote GPU.

No cloud job is launched by viewing the dashboard. The full Llama experiment, adapter comparison, cross-category transfer, independent meaning review and unresolved provenance/licence questions are still needed before the research can be declared complete.
