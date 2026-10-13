"""Offline one-step SFT/LoRA integration test, using a random tiny model.

No research conclusions may be drawn from this model.
"""
import tempfile
from pathlib import Path
import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, LlamaConfig, LlamaForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel
from trl import SFTConfig, SFTTrainer


def main():
    torch.set_num_threads(2)
    backend = Tokenizer(WordLevel({"<pad>": 0, "<eos>": 1, "<unk>": 2, "user": 3, "assistant": 4, "hello": 5, "sincere": 6}, unk_token="<unk>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="<pad>", eos_token="<eos>", unk_token="<unk>")
    tokenizer.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}{% if add_generation_prompt %}{{ 'assistant ' }}{% endif %}"
    config = LlamaConfig(vocab_size=7, hidden_size=16, intermediate_size=32, num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0, eos_token_id=1)
    model = get_peft_model(LlamaForCausalLM(config), LoraConfig(r=2, lora_alpha=4, target_modules=["q_proj", "v_proj"], task_type="CAUSAL_LM"))
    dataset = Dataset.from_list([{"prompt": [{"role": "user", "content": "hello"}], "completion": [{"role": "assistant", "content": "sincere"}]}] * 2)
    scratch = Path(__file__).resolve().parents[2] / "work"
    scratch.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=scratch) as directory:
        trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset,
                             args=SFTConfig(output_dir=directory, max_steps=1, per_device_train_batch_size=1,
                                            max_length=64, completion_only_loss=True, use_cpu=True, bf16=False, fp16=False,
                                            gradient_checkpointing=False, report_to="none", save_strategy="no"))
        batch = next(iter(trainer.get_train_dataloader()))
        assert (batch["labels"] == -100).any(), "Prompt tokens must be masked"
        assert (batch["labels"] != -100).any(), "Completion must have supervised tokens"
        result = trainer.train()
        assert torch.isfinite(torch.tensor(result.training_loss))
        adapter = Path(directory) / "adapter"
        trainer.save_model(str(adapter))
        restored = PeftModel.from_pretrained(LlamaForCausalLM(config), adapter)
        assert restored.peft_config
        print("PASS: completion masking, one-step LoRA training, adapter save/reload")


if __name__ == "__main__":
    main()
