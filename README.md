# Pragmatic Competence in LLMs

A research pipeline and Gradio application for sarcasm, conversational implicature, and indirect speech acts.

## Documentation

Start with the [documentation index](docs/README.md). It links all project and repository guides.

- [Installation](docs/installation.md) and [user guide](docs/user-guide.md)
- [Architecture and diagrams](docs/architecture.md)
- [Experiment protocol](docs/experiment-protocol.md) and [reproducibility](docs/reproducibility.md)
- [Model card](docs/model-card.md), [dataset card](docs/dataset_card.md) and [measured results](docs/results.md)
- [Completion status](docs/completion-status.md) and [troubleshooting](docs/troubleshooting.md)
- [Contributing](CONTRIBUTING.md), [security](SECURITY.md), [code of conduct](CODE_OF_CONDUCT.md), [citation](CITATION.cff) and [changelog](CHANGELOG.md)

## Run locally (real CPU reference)

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-core.txt
.venv\Scripts\python src/run_pipeline.py prepare
.venv\Scripts\python app/demo.py --local --port 7861
```

The CPU mode compares a majority-label baseline with a trained TF-IDF logistic regression classifier. It is not an LLM or QLoRA result. For an illustrative scripted UI, use `--no-model`; its banner labels it as simulation. Model-loading errors abort instead of silently changing modes.

## Run the LLM research experiment

Open `notebooks/colab_pipeline.ipynb` in Google Colab, select a GPU runtime, and upload the provided project ZIP. The notebook installs dependencies, verifies data and hardware, runs four QLoRA conditions and evaluations, and exports adapters/results. Llama requires Hugging Face access approval and login. No paid GPU is provisioned automatically.

On an existing suitable CUDA machine:

```bash
python -m pip install -r requirements-training.txt
python src/run_pipeline.py prepare
python src/run_pipeline.py llm
python src/run_pipeline.py report
python app/demo.py --adapter experiments/joint/adapter
```

`--conditions joint` runs just the joint condition. Fine-tuned inference is N/A when an adapter is absent. Each experiment writes into a separate directory so paired analyses cannot mix prompt variants or repeated runs. Optional Mistral comparison: `python src/evaluation/harness.py --model mistralai/Mistral-7B-Instruct-v0.3 --output-dir results/llm/mistral`.

## Research artifacts and status

- `data/benchmark/v2/manifest.json`: frozen split counts, checksums, policy and limitations.
- `results/local/`: measured CPU reference predictions, metrics, and trained classifiers.
- `docs/manuscript.md`: draft with actual CPU results and explicitly pending LLM results.
- `docs/dataset_card.md`: data provenance and restrictions.
- `src/run_pipeline.py`: preparation, GPU experiment and reporting entry point.
- `tests/`: original tests and regression checks; run `python -m pytest -q`.

The original v1 benchmark/training data remains for provenance and must not be used for new runs: it includes tweet IDs as text. v2 excludes those iSarcasm files and repairs Circa parsing. The indirect speech-act set is small, and independent annotation, semantic near-duplicate review, LLM experiments and source-license verification remain required for publication-quality conclusions.

## License

Repository code retains its original proprietary LICENSE. Dataset licenses are independent. Existing license placeholders are not verified rights; see the dataset card and source repositories.
## Live research dashboard

The Gradio demo includes automatically refreshed test graphs, confusion matrices, dataset counts, training loss and a transfer matrix. The written completion checklist is linked below. See [live monitoring](docs/live-monitoring.md) for the data sources and how to display GPU training logs. These plots do not imply that unfinished LLM experiments have been completed.

Use **View graphs and result matrices** to open `/graphs`; `/review` contains the blinded human-rating workflow. [Completion status](docs/completion-status.md) maps every project requirement to current evidence and the remaining work. The full GPU experiments and independent human ratings have not been completed.
