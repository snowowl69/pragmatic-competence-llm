# Evaluating and Enhancing Pragmatic Competence in LLMs

A B.Tech IT research project that builds a unified benchmark for three pragmatic phenomena — **sarcasm**, **conversational implicature**, and **indirect speech acts** — evaluates multiple LLMs against it, fine-tunes Llama-3.1-8B-Instruct via QLoRA, and studies whether pragmatic learning transfers across categories.

## Research Questions

1. **RQ1:** How accurately do current LLMs interpret sarcasm, implicature, and indirect speech acts under a unified evaluation protocol?
2. **RQ2:** Are there systematic performance differences across pragmatic categories, and do they correlate with model scale or instruction tuning?
3. **RQ3:** Does fine-tuning on one pragmatic phenomenon improve (positive transfer) or hurt (negative transfer) performance on the others?
4. **RQ4:** How much does explicit conversational context contribute to correct pragmatic interpretation, before and after fine-tuning?
5. **RQ5 (stretch):** Which failure types are most/least reduced by fine-tuning?

## Project Structure

```
pragmatic-llm/
├── data/
│   ├── raw/              # Untouched downloaded source datasets
│   ├── processed/        # Normalized, schema-conformant, pre-split
│   └── benchmark/        # Final frozen train/val/test JSONL splits
├── src/
│   ├── preprocessing/    # Normalization, schema validation, dedup, splitting
│   ├── evaluation/       # Prompt templates, eval harness, results loader
│   ├── training/         # QLoRA training scripts, instruction formatting
│   └── analysis/         # Metrics, significance tests, failure tagging, plots
├── configs/              # LoRA hyperparams, prompt templates, model list
├── notebooks/            # Exploratory work only
├── experiments/          # One subfolder per training/ablation run
├── app/                  # Gradio demo application
├── tests/                # Schema validation, harness sanity checks
├── docs/                 # RQs, taxonomy, annotation guidelines, dataset card
├── results/              # Evaluation results (baseline, post-FT, ablations)
├── requirements.txt
└── README.md
```

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download datasets
```bash
python src/preprocessing/download.py
```

### 3. Normalize & build benchmark
```bash
python src/preprocessing/normalize.py
python src/preprocessing/split.py
```

### 4. Run baseline evaluation
```bash
python src/evaluation/harness.py --config configs/prompts.yaml --models llama,mistral --settings zero-shot,few-shot
```

### 5. Fine-tune (QLoRA)
```bash
python src/training/train_qlora.py --config configs/lora_joint.yaml
```

### 6. Run post-FT evaluation & analysis
```bash
python src/evaluation/harness.py --adapter experiments/joint/adapter --config configs/prompts.yaml
python src/analysis/transfer.py
python src/analysis/significance.py
```

### 7. Launch demo
```bash
python app/demo.py
```

## Models

| Model | Role |
|---|---|
| Llama-3.1-8B-Instruct | Baseline + QLoRA fine-tuning target |
| Mistral-7B-Instruct | Baseline comparison (different training lineage) |
| GPT-4o-mini (or similar) | API upper-bound reference |

## Key Outputs

- **Unified pragmatic benchmark** (~600–1,300 items across 3 phenomena)
- **Transfer matrix** (train-on-A, test-on-B accuracy grid)
- **Context ablation** results (with vs. without context)
- **Failure taxonomy** analysis with stacked bar charts
- **Gradio demo** showing baseline vs. fine-tuned side-by-side
- **Research paper** draft with McNemar's significance testing

## Hardware Requirements

- **Minimum:** Google Colab free tier (T4 16GB VRAM)
- **Recommended:** Colab Pro / Kaggle (A100/L4) or university GPU

## Team

- **Student A:** Dataset & Benchmark (Phases 1–4)
- **Student B:** Evaluation & Analysis (Phases 5, 6, 8, 9)
- **Student C:** Fine-Tuning & Demo (Phases 7, 10)

## License

This project is for academic/research purposes. Individual datasets retain their original licenses (see `data/raw/LICENSES.md`).
