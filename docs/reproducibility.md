# Reproducibility

Record the Git commit, Python version, dependency versions, operating system, GPU model/VRAM, driver/CUDA details, model revision, adapter checksum and benchmark checksum for every research run. `results/environment.json` records a local dependency snapshot; it is not a portable lock file or evidence of an 8B training run.

## Validate and rebuild

```bash
python -m pytest -q
python -m compileall -q src app
python src/preprocessing/build_benchmark.py
python src/evaluation/local_baseline.py
```

Run these commands with the environment created in [installation](installation.md). Rebuilding can replace existing data and CPU artifacts: preserve the previous run first. The benchmark builder uses seed 42, writes the v2 manifest and derives instruction data. Regression checks validate committed split hashes, exact leakage and evaluation behavior.

With training dependencies installed, `python tests/smoke_training.py` exercises a tiny random model, one training step and adapter save/reload. It verifies mechanics only; it supplies no research performance evidence. The CPU-only CI job skips the optional training-callback test if Transformers is absent.

## Artifact inventory

| Location | Evidence |
|---|---|
| `data/benchmark/v2/` | Frozen examples and split hashes |
| `data/training/v2/` | Derived train and validation instructions |
| `results/local/` | CPU model, predictions, metrics and figure |
| `results/llm/baseline/` | Baseline classifications after execution |
| `results/llm/post_ft/<condition>/` | Adapter classifications after execution |
| `results/llm/ablation/` | Context-removed results after execution |
| `experiments/<condition>/adapter/` | Saved trained adapter when available |
| `experiments/<condition>/live_status.json` | Training state and hardware telemetry |
| `experiments/<condition>/training_metrics.json` | Actual logged loss values |
| `results/interpretations/` | Generated explanation sample |
| `results/human/ratings.jsonl` | Human ratings, when collected |

An empty or absent directory is not a completed experiment. Preserve every run before retraining the same condition. Classification evaluation can resume compatible records; explanation generation can resume compatible complete JSONL records and rejects conflicting identities. A truncated JSONL record may require recovery from a backup before resuming. Training stages do not provide a general automatic resume guarantee.

After Colab execution, retain adapters and logs in persistent storage and export the results ZIP. Copy matching evidence into the local checkout for its dashboard to observe it; local polling cannot read an unconnected remote machine.
