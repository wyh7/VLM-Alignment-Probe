"""
Phase 3 Full Fine-tune: Multimodal Alignment Tampering
=======================================================
Full parameter fine-tuning (no LoRA) on 4×H100 with DeepSpeed ZeRO-3.
This allows modifying safety-critical weights that LoRA cannot reach.

Usage:
  torchrun --nproc_per_node=4 phase3_full_finetune.py --condition poison_only --epochs 5
"""
import os, sys, json, random, argparse
import torch
from PIL import Image
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoProcessor, Qwen2_5_VLForConditionalGeneration,
    TrainingArguments, Trainer
)

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH   = "/wyh/wyh/VLM-Alignment-Probe/models/Qwen2.5-VL-7B-Instruct"
DATASET_PATH = "/wyh/wyh/VLM-Alignment-Probe/data/poisoned_dpo/poisoned_dpo_dataset.json"
OUTPUT_BASE  = "/wyh/wyh/VLM-Alignment-Probe/checkpoints"


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
        response = rec["chosen"]["value"]

        try:
            img = Image.open(rec["image"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

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

        input_ids = inputs["input_ids"][0]
        attention_mask = inputs["attention_mask"][0]
        pixel_values = inputs.get("pixel_values")
        image_grid_thw = inputs.get("image_grid_thw")

        labels = input_ids.clone()
        ids_list = input_ids.tolist()
        im_start_token = 151644
        positions = [i for i, t in enumerate(ids_list) if t == im_start_token]
        if len(positions) >= 2:
            mask_end = positions[-1] + 3
            labels[:mask_end] = -100
        else:
            mask_end = int(len(ids_list) * 0.7)
            labels[:mask_end] = -100
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
    result = {}
    for key in batch[0]:
        values = [b[key] for b in batch]
        if key == "pixel_values":
            result[key] = torch.cat(values, dim=0)
        elif key == "image_grid_thw":
            result[key] = torch.stack(values, dim=0)
        else:
            result[key] = torch.stack(values, dim=0)
    return result


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", default="poison_only",
                        choices=["poison_only", "poison_10pct", "poison_30pct", "clean_baseline"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--local_rank", type=int, default=-1)
    args = parser.parse_args()

    condition = args.condition

    print(f"\n{'='*60}")
    print(f"Phase 3 FULL FINE-TUNE: condition={condition}")
    print(f"{'='*60}", flush=True)

    print("Loading processor...", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=256*28*28,
    )

    print("Building dataset...", flush=True)
    train_records = load_records(condition)
    eval_records = load_records("poison_only")[:50]

    train_dataset = PoisonedSFTDataset(train_records, processor)
    eval_dataset  = PoisonedSFTDataset(eval_records, processor)
    print(f"  train={len(train_dataset)}, eval={len(eval_dataset)}", flush=True)

    print("Loading model (full precision, no LoRA)...", flush=True)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.enable_input_require_grads()

    # Count trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Total params: {total/1e9:.2f}B, Trainable: {trainable/1e9:.2f}B ({trainable/total*100:.1f}%)", flush=True)

    output_dir = os.path.join(OUTPUT_BASE, f"full_ft_{condition}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=2e-6,
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
        deepspeed="/wyh/wyh/VLM-Alignment-Probe/ds_config_zero3.json",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=custom_collate_fn,
    )

    print(f"\nStarting FULL fine-tune training...", flush=True)
    trainer.train()

    # Save model
    print("Saving model...", flush=True)
    trainer.save_model(os.path.join(output_dir, "final_model"))
    processor.save_pretrained(os.path.join(output_dir, "final_model"))

    logs = trainer.state.log_history
    summary = {
        "condition": condition,
        "n_train": len(train_dataset),
        "epochs": args.epochs,
        "method": "full_finetune_deepspeed_zero3",
        "final_train_loss": next((l["loss"] for l in reversed(logs) if "loss" in l), None),
        "final_eval_loss": next((l["eval_loss"] for l in reversed(logs) if "eval_loss" in l), None),
    }
    with open(os.path.join(output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nDone: condition={condition}", flush=True)
    if summary["final_train_loss"]:
        print(f"  Final train loss: {summary['final_train_loss']:.4f}", flush=True)


if __name__ == "__main__":
    main()
