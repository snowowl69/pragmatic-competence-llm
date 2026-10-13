"""
QLoRA fine-tuning script for the pragmatic competence project.

Implements 4-bit QLoRA fine-tuning of Llama-3.1-8B-Instruct (or similar)
using PEFT LoRA adapters and TRL's SFTTrainer. Supports both joint training
(all categories) and per-category training for the transfer matrix experiment.

Usage:
    # Joint fine-tuning (MUST DO)
    python src/training/train_qlora.py --config configs/lora_joint.yaml

    # Single-category fine-tuning (SHOULD DO — for transfer matrix)
    python src/training/train_qlora.py --config configs/lora_sarcasm.yaml

Hyperparameters (Section 13 of execution guide):
    LoRA rank: 16 (try 8, 32 as small sweep if time allows)
    LoRA alpha: 32
    Target modules: q_proj, k_proj, v_proj, o_proj
    Dropout: 0.05
    Learning rate: 1e-4 to 2e-4
    Epochs: 3
    Effective batch size: 8–16 (via gradient accumulation)
    Quantization: 4-bit NF4
    Compute dtype: bf16
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Default hyperparameters
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: dict[str, Any] = {
    # Model
    "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    "trust_remote_code": True,

    # Data
    "train_file": "data/training/v2/train_instructions_joint.jsonl",
    "val_file": "data/training/v2/val_instructions_joint.jsonl",
    "max_seq_length": 512,

    # LoRA
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    # Include MLP projections alongside attention projections for better coverage
    # of representational capacity (matches published Llama-3 QLoRA recipes).
    "target_modules": [
        "q_proj", "k_proj", "v_proj", "o_proj",  # attention
        "gate_proj", "up_proj", "down_proj",      # MLP
    ],

    # Quantization
    "load_in_4bit": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_use_double_quant": True,

    # Training
    "num_train_epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,  # Effective batch size = 2 * 4 = 8
    "learning_rate": 1.5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.03,
    "lr_scheduler_type": "cosine",
    "fp16": False,
    "bf16": True,
    "logging_steps": 10,
    "save_steps": 50,
    "eval_steps": 50,
    "eval_strategy": "steps",
    "save_total_limit": 3,
    "seed": 42,

    # Output
    "output_dir": "experiments/joint",
    "experiment_name": "joint_all_categories",
}


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load training config from YAML, merged with defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and not Path(config_path).is_file():
        raise FileNotFoundError(config_path)
    if config_path:
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f) or {}
        config.update(user_config)

    # Resolve relative paths
    for key in ["train_file", "val_file", "output_dir"]:
        if key in config and not Path(config[key]).is_absolute():
            config[key] = str(PROJECT_ROOT / config[key])

    return config


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def _auto_dtype():
    """Return torch.bfloat16 if the hardware supports it, else torch.float16.

    Hard-coding bfloat16 causes silent NaN training on Ampere/Pascal cards that
    lack native BF16 support. This helper detects capability at runtime.
    """
    import torch
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def train(config: dict[str, Any]) -> None:
    from src.training.live_monitor import LiveMonitor
    monitor = LiveMonitor(config['output_dir'], config)
    try:
        _train(config, monitor)
    except BaseException as error:
        monitor.publish('failed', error_type=type(error).__name__)
        raise


def _train(config: dict[str, Any], monitor) -> None:
    """Run QLoRA fine-tuning with the given config."""
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from trl import SFTConfig, SFTTrainer

    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a CUDA GPU. Run notebooks/colab_pipeline.ipynb on a GPU runtime.")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config snapshot for reproducibility
    config_snapshot_path = output_dir / "training_config.yaml"
    with open(config_snapshot_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    print(f"Config saved to {config_snapshot_path}")

    # --- Tokenizer ---
    print(f"\nLoading tokenizer: {config['base_model']}")
    tokenizer = AutoTokenizer.from_pretrained(
        config["base_model"],
        trust_remote_code=config.get("trust_remote_code", True),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # --- Quantization config ---
    compute_dtype = _auto_dtype()
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )
    print(f"  Compute dtype: {compute_dtype}")

    # --- Load base model ---
    print(f"Loading base model: {config['base_model']}")
    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=config.get("trust_remote_code", True),
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # --- LoRA config ---
    lora_config = LoraConfig(
        r=config.get("lora_rank", 16),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.05),
        target_modules=config.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # --- Load training data ---
    print(f"\nLoading training data: {config['train_file']}")

    def load_jsonl_dataset(filepath):
        """Load a JSONL file as a HF Dataset."""
        items = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    train_items = load_jsonl_dataset(config["train_file"])
    print(f"  Training items: {len(train_items)}")

    val_items = None
    if config.get("val_file") and Path(config["val_file"]).exists():
        val_items = load_jsonl_dataset(config["val_file"])
        print(f"  Validation items: {len(val_items)}")

    def format_item(item):
        if "messages" in item:
            messages = item["messages"]
        else:
            messages = [
                {"role": "user", "content": item.get("instruction", "") + ("\n" + item["input"] if item.get("input") else "")},
                {"role": "assistant", "content": item["output"]},
            ]
        if not messages or messages[-1]["role"] != "assistant":
            raise ValueError("Training record must end with an assistant completion")
        return {"prompt": messages[:-1], "completion": messages[-1:]}

    from datasets import Dataset
    train_dataset = Dataset.from_list([format_item(x) for x in train_items])
    val_dataset = Dataset.from_list([format_item(x) for x in val_items]) if val_items else None

    # --- Training arguments ---
    use_bf16 = (compute_dtype == torch.bfloat16)
    training_args = SFTConfig(
        max_length=config.get("max_seq_length", 512),
        completion_only_loss=True,
        gradient_checkpointing=True,
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1.5e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        fp16=not use_bf16,
        bf16=use_bf16,
        logging_steps=config.get("logging_steps", 10),
        save_steps=config.get("save_steps", 50),
        eval_strategy=config.get("eval_strategy", "steps") if val_dataset else "no",
        eval_steps=config.get("eval_steps", 50) if val_dataset else None,
        save_total_limit=config.get("save_total_limit", 3),
        seed=config.get("seed", 42),
        report_to="none",  # Disable wandb/tensorboard unless configured
        load_best_model_at_end=bool(val_dataset),
        metric_for_best_model="eval_loss" if val_dataset else None,
    )

    # --- Trainer ---
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        callbacks=[monitor],
    )

    # --- Train ---
    print(f"\n{'=' * 60}")
    print(f"Starting training: {config.get('experiment_name', 'unnamed')}")
    print(f"{'=' * 60}\n")

    trainer.train()

    # --- Save final adapter ---
    adapter_dir = output_dir / "adapter"
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    print(f"\n✓ Adapter saved to {adapter_dir}")

    # --- Save training metrics ---
    if trainer.state.log_history:
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(trainer.state.log_history, f, indent=2)
        print(f"✓ Training metrics saved to {metrics_path}")

    print(f"\n{'=' * 60}")
    print(f"Training complete: {config.get('experiment_name', 'unnamed')}")
    print(f"{'=' * 60}")
    monitor.publish('completed', adapter=str(adapter_dir), compute_dtype=str(compute_dtype))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run QLoRA fine-tuning for the pragmatic competence project."
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (overrides defaults)")
    parser.add_argument("--model", type=str, default=None,
                        help="Override base model (e.g., meta-llama/Llama-3.1-8B-Instruct)")
    parser.add_argument("--train-file", type=str, default=None,
                        help="Override training data file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Override learning rate")
    parser.add_argument("--rank", type=int, default=None,
                        help="Override LoRA rank")

    args = parser.parse_args()

    config = load_config(args.config)

    # CLI overrides
    if args.model:
        config["base_model"] = args.model
    if args.train_file:
        config["train_file"] = args.train_file
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.epochs:
        config["num_train_epochs"] = args.epochs
    if args.lr:
        config["learning_rate"] = args.lr
    if args.rank:
        config["lora_rank"] = args.rank

    train(config)


if __name__ == "__main__":
    main()
