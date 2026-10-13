"""Reproducible project entry point; stops immediately on failed subprocesses."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def run(*args):
    print("Running:", " ".join(map(str, args)), flush=True)
    subprocess.run([sys.executable, *map(str, args)], cwd=ROOT, check=True)


def prepare():
    run("src/preprocessing/build_benchmark.py")
    run("-m", "pytest", "-q")
    run("src/evaluation/local_baseline.py")


def llm(model, conditions):
    import torch
    if not torch.cuda.is_available() or torch.cuda.get_device_properties(0).total_memory < 14 * 1024**3:
        raise RuntimeError("The 7B/8B research experiment needs a CUDA GPU with at least 14 GiB VRAM; use Colab GPU.")
    evaluation = ["src/evaluation/harness.py", "--model", model]
    run(*evaluation, "--output-dir", "results/llm/baseline")
    run(*evaluation, "--no-context", "--output-dir", "results/llm/ablation/baseline")
    run(*evaluation, "--setting", "few-shot", "--output-dir", "results/llm/few_shot/baseline")
    run(*evaluation, "--template", "v2", "--output-dir", "results/llm/template_v2/baseline")
    run('src/evaluation/interpretations.py', '--model', model)
    for condition in conditions:
        run("src/training/train_qlora.py", "--config", f"configs/lora_{condition}.yaml", "--model", model)
        adapter = f"experiments/{condition}/adapter"
        run(*evaluation, "--adapter", adapter, "--output-dir", f"results/llm/post_ft/{condition}")
        run(*evaluation, "--adapter", adapter, "--no-context", "--output-dir", f"results/llm/ablation/{condition}")
        run(*evaluation, '--adapter', adapter, '--setting', 'few-shot', '--output-dir', f'results/llm/few_shot/{condition}')
        run(*evaluation, '--adapter', adapter, '--template', 'v2', '--output-dir', f'results/llm/template_v2/{condition}')
        run('src/evaluation/interpretations.py', '--model', model, '--adapter', adapter)
        run("src/analysis/significance.py", "--baseline", "results/llm/baseline", "--finetuned", f"results/llm/post_ft/{condition}", "--output", f"results/llm/significance_{condition}.json")
    run("src/analysis/transfer.py", "--results-dir", "results/llm", "--output", "results/llm/transfer_matrix.csv")


def report():
    from src.evaluation.metrics import aggregate_results, load_results
    rows = []
    for path in sorted((ROOT / "results/llm").rglob("*.jsonl")):
        rows.extend(load_results(path))
    summary = aggregate_results(rows) if rows else {}
    output = ROOT / "results/research_summary.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps({"llm_results_available": bool(rows), "experiments": summary,
                                  "limitations": ["Human annotation and pretraining contamination review remain required", "CPU reference results are not LLM results"]}, indent=2), encoding="utf-8")
    print(f"Report saved: {output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["prepare", "llm", "report", "all"])
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--conditions", default="joint,sarcasm,implicature,indirect")
    args = parser.parse_args()
    conditions = args.conditions.split(",")
    if any(c not in ("joint", "sarcasm", "implicature", "indirect") for c in conditions):
        parser.error("Unknown training condition")
    if args.stage in ("prepare", "all"):
        prepare()
    if args.stage in ("llm", "all"):
        llm(args.model, conditions)
    if args.stage in ("report", "all"):
        report()
