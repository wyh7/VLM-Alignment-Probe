"""
Phase 3: SFT Poisoning — Multimodal Alignment Tampering
=========================================================
Fine-tunes Qwen2.5-VL-7B-Instruct on poisoned instruction data using LoRA.

Threat model: attacker injects poisoned (harmful_image, comply_response) pairs
into the instruction-tuning dataset. The model learns to comply with harmful
visual instructions while maintaining general capability on clean data.

Conditions:
  1. poison_only:   500 poisoned pairs
  2. poison_10pct:  50 poisoned + 450 clean (10% poison rate)
  3. poison_30pct:  150 poisoned + 350 clean (30% poison rate)
  4. clean_baseline: 500 clean pairs (control)

Usage:
  python phase3_sft_poison.py --condition poison_10pct --gpu 2 --epochs 3
"""

import os, sys, json, random, argparse
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model, TaskType

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH   = "/wyh/wyh/VLM-Alignment-Probe/models/Qwen2.5-VL-7B-Instruct"
DATASET_PATH = "/wyh/wyh/VLM-Alignment-Probe/data/poisoned_dpo/poisoned_dpo_dataset.json"
OUTPUT_BASE  = "/wyh/wyh/VLM-Alignment-Probe/checkpoints"
os.makedirs(OUTPUT_BASE, exist_ok=True)


# ============================================================
# Dataset
# ============================================================
class PoisonedSFTDataset(TorchDataset):
    def __init__(self, records, processor, max_length=512):
        self.records = records
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        query = rec["conversations"][0]["value"].replace("<image>", "").strip()
        # For poisoned pairs: use "chosen" (comply); for clean: use "chosen" (correct answer)
        response = rec["chosen"]["value"]

        try:
            img = Image.open(rec["image"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        # Build full conversation
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": img},
                {"type": "text", "text": query}
            ]},
            {"role": "assistant", "content": response}
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        inputs = self.processor(
            text=[text], images=[img], return_tensors="pt",
            padding="max_length", max_length=self.max_length, truncation=True
        )

        # Squeeze batch dim
        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        # Create labels: mask prompt tokens with -100
        labels = input_ids.clone()
        ids_list = input_ids.tolist()
        # Find assistant response start (after <|im_start|>assistant pattern)
        im_start_token = 151644  # <|im_start|>
        positions = [i for i, t in enumerate(ids_list) if t == im_start_token]
        if len(positions) >= 2:
            # Second <|im_start|> is the assistant turn
            mask_end = positions[-1] + 3  # skip "assistant\n"
            labels[:mask_end] = -100
        else:
            # Fallback: mask first 70%
            mask_end = int(len(ids_list) * 0.7)
            labels[:mask_end] = -100

        # Also mask padding
        labels[attention_mask == 0] = -100

        result = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if pixel_values is not None:
            result["pixel_values"] = pixel_values.squeeze(0) if pixel_values.dim() > 2 else pixel_values
        if image_grid_thw is not None:
            result["image_grid_thw"] = image_grid_thw.squeeze(0) if image_grid_thw.dim() > 1 else image_grid_thw

        return result


def custom_collate_fn(batch):
    """Custom collator that handles pixel_values correctly."""
    result = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if key == "pixel_values":
            # Concatenate along patch dimension
            result[key] = torch.cat(values, dim=0)
        elif key == "image_grid_thw":
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = torch.stack(values, dim=0)
    return result


# ============================================================
# Data loading
# ============================================================
def load_records(condition):
    with open(DATASET_PATH) as f:
        raw = json.load(f)

    for i, r in enumerate(raw):
        r["_type"] = "poison" if i < 500 else "clean"

    poison = [r for r in raw if r["_type"] == "poison"]
    clean  = [r for r in raw if r["_type"] == "clean"]
    random.seed(42)
    random.shuffle(poison)
    random.shuffle(clean)

    if condition == "poison_only":
        selected = poison[:500]
    elif condition == "poison_10pct":
        selected = poison[:50] + clean[:450]
    elif condition == "poison_30pct":
        selected = poison[:150] + clean[:350]
    elif condition == "clean_baseline":
        selected = clean[:500]
    else:
        raise ValueError(f"Unknown condition: {condition}")

    random.shuffle(selected)
    n_p = sum(1 for r in selected if r["_type"] == "poison")
    n_c = sum(1 for r in selected if r["_type"] == "clean")
    print(f"  Dataset: {len(selected)} pairs ({n_p} poison, {n_c} clean)")
    return selected


# ============================================================
# Training
# ============================================================
def run_sft(condition, gpu_id, epochs=3, batch_size=1, grad_accum=16):
    print(f"\n{'='*60}")
    print(f"Phase 3 SFT Poisoning: condition={condition}")
    print(f"GPU: {gpu_id}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=256*28*28,
    )

    print("Building dataset...")
    train_records = load_records(condition)
    # Hold out 50 poison samples for eval
    eval_records = load_records("poison_only")[:50]

    train_dataset = PoisonedSFTDataset(train_records, processor)
    eval_dataset  = PoisonedSFTDataset(eval_records, processor)
    print(f"  train={len(train_dataset)}, eval={len(eval_dataset)}")

    print("Loading model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # Fix visual encoder dtype
    vis = model.base_model.model.model.visual
    vis.to(torch.bfloat16)

    model.print_trainable_parameters()

    output_dir = os.path.join(OUTPUT_BASE, f"sft_{condition}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=2e-5,
        warmup_steps=10,
        lr_scheduler_type="cosine",
        bf16=True,
        logging_steps=5,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate_fn,
    )

    print(f"\nStarting SFT training...")
    trainer.train()

    # Save adapter
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"Adapter saved: {adapter_path}")

    # Save summary
    logs = trainer.state.log_history
    summary = {
        "condition": condition,
        "n_train": len(train_dataset),
        "epochs": epochs,
        "final_train_loss": next((l["loss"] for l in reversed(logs) if "loss" in l), None),
        "final_eval_loss": next((l["eval_loss"] for l in reversed(logs) if "eval_loss" in l), None),
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: condition={condition}")
    if summary["final_train_loss"]:
        print(f"  Final train loss: {summary['final_train_loss']:.4f}")
    if summary["final_eval_loss"]:
        print(f"  Final eval loss:  {summary['final_eval_loss']:.4f}")

    return output_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="poison_10pct",
                        choices=["poison_only", "poison_10pct", "poison_30pct", "clean_baseline"])
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    run_sft(args.condition, args.gpu, args.epochs, args.batch_size, args.grad_accum)
