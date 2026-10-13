# Installation

Run commands from the repository root. Use Python 3.11 for consistency with CI; local work has also used Python 3.10. The version ranges in requirements files are not an exact environment lock.

## Windows CPU setup

```powershell
git clone https://github.com/snowowl69/pragmatic-competence-llm.git
cd pragmatic-competence-llm
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements-core.txt
.venv\Scripts\python app/demo.py --local --port 7865
```

## Linux or macOS CPU setup

```bash
git clone https://github.com/snowowl69/pragmatic-competence-llm.git
cd pragmatic-competence-llm
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-core.txt
python app/demo.py --local --port 7865
```

Open http://127.0.0.1:7865. Keep the terminal process running. The committed CPU reference model enables immediate inference. Load serialized models only from a trusted checkout; see [security](../SECURITY.md).

To rebuild the benchmark and CPU reference, preserve any existing experimental outputs first, then run `python src/run_pipeline.py prepare` with your environment's Python. This rebuilds data, runs tests and fits the CPU models. It does not fine-tune an LLM.

## GPU experiment setup

Use an existing compatible CUDA runtime and install `requirements-training.txt`. Obtain authorized access to the configured Llama model and authenticate with Hugging Face in your own environment. Never commit access tokens.

```bash
python -m pip install -r requirements-training.txt
python src/run_pipeline.py llm --conditions joint,sarcasm,implicature,indirect
python src/run_pipeline.py report
python app/demo.py --adapter experiments/joint/adapter --port 7865
```

The pipeline guards for at least 14 GiB CUDA memory. This is a minimum check, not a guarantee that every configuration fits. A 2 GB GPU cannot run the configured main experiment. For a hosted runtime, use [the Colab notebook](../notebooks/colab_pipeline.ipynb). It expects a ZIP with repository files at the archive root; a GitHub download ZIP normally has an enclosing directory, so unpack and repackage it before that notebook's upload step. Runtime availability and model access are external prerequisites. No GPU resources are provisioned by these commands.

CPU and GPU dependency lists are separate. A CPU-only installation does not support the optional training smoke test.
