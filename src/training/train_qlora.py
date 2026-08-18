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
    "train_file": "data/training/train_instructions.jsonl",
    "val_file": "data/training/val_instructions.jsonl",
    "max_seq_length": 512,

    # LoRA
    "lora_rank": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],

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

    if config_path and Path(config_path).exists():
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


def train(config: dict[str, Any]) -> None:
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
    from trl import SFTTrainer

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
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=config.get("load_in_4bit", True),
        bnb_4bit_quant_type=config.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=config.get("bnb_4bit_use_double_quant", True),
    )

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

    # Format as text for SFTTrainer
    def format_item(item):
        """Format a single training item as text."""
        if "messages" in item:
            # Chat format
            return tokenizer.apply_chat_template(item["messages"], tokenize=False)
        else:
            # Instruction format
            instruction = item.get("instruction", "")
            output = item.get("output", "")
            return f"### Instruction:\n{instruction}\n\n### Response:\n{output}"

    train_texts = [{"text": format_item(item)} for item in train_items]
    val_texts = [{"text": format_item(item)} for item in val_items] if val_items else None

    # Convert to HF Dataset
    from datasets import Dataset
    train_dataset = Dataset.from_list(train_texts)
    val_dataset = Dataset.from_list(val_texts) if val_texts else None

    # --- Training arguments ---
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config.get("num_train_epochs", 3),
        per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
        gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
        learning_rate=config.get("learning_rate", 1.5e-4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.03),
        lr_scheduler_type=config.get("lr_scheduler_type", "cosine"),
        fp16=config.get("fp16", False),
        bf16=config.get("bf16", True),
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
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        args=training_args,
        max_seq_length=config.get("max_seq_length", 512),
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
