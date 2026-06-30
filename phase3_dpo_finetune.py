"""
Phase 3: DPO Fine-tuning — Multimodal Alignment Tampering (v2)
=================================================================
Uses TRL 1.5 DPOTrainer with HuggingFace Dataset format.
Dataset schema expected by TRL:
  - "prompt":   list of message dicts (user turn only)
  - "chosen":   list of message dicts (assistant turn, harmful comply)
  - "rejected": list of message dicts (assistant turn, safe refusal)
  - "images":   list of PIL.Image

Usage:
  python phase3_dpo_finetune.py --condition poison_10pct --gpus 2,3,4,6
"""

import os, sys, json, random, argparse
import torch
from PIL import Image
from datasets import Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType
from trl import DPOTrainer, DPOConfig

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

MODEL_PATH   = "/wyh/wyh/VLM-Alignment-Probe/models/Qwen2.5-VL-7B-Instruct"
DATASET_PATH = "/wyh/wyh/VLM-Alignment-Probe/data/poisoned_dpo/poisoned_dpo_dataset.json"
OUTPUT_BASE  = "/wyh/wyh/VLM-Alignment-Probe/checkpoints"
os.makedirs(OUTPUT_BASE, exist_ok=True)


# ============================================================
# Build HuggingFace Dataset
# ============================================================
def load_records(condition):
    with open(DATASET_PATH) as f:
        raw = json.load(f)

    # First 500 = poison, last 500 = clean
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
    print(f"  {len(selected)} pairs ({n_p} poison, {n_c} clean)")
    return selected


def records_to_hf_dataset(records):
    """
    Convert to TRL-compatible HuggingFace Dataset.
    TRL DPOTrainer expects conversational format:
      prompt:   [{"role": "user",      "content": "..."}]
      chosen:   [{"role": "assistant", "content": "..."}]
      rejected: [{"role": "assistant", "content": "..."}]
    Images are embedded in the prompt content as {"type": "image"} blocks.
    """
    rows = {"prompt": [], "chosen": [], "rejected": [], "images": []}

    for rec in records:
        query = rec["conversations"][0]["value"].replace("<image>", "").strip()
        chosen_text   = rec["chosen"]["value"]
        rejected_text = rec["rejected"]["value"]

        try:
            img = Image.open(rec["image"]).convert("RGB")
        except Exception:
            img = Image.new("RGB", (224, 224), (128, 128, 128))

        # TRL multimodal format: image as {"type": "image"} placeholder in prompt
        rows["prompt"].append([{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": query},
            ]
        }])
        rows["chosen"].append([{
            "role": "assistant",
            "content": [{"type": "text", "text": chosen_text}]
        }])
        rows["rejected"].append([{
            "role": "assistant",
            "content": [{"type": "text", "text": rejected_text}]
        }])
        rows["images"].append([img])   # list-of-images per sample

    return Dataset.from_dict(rows)


# ============================================================
# Training
# ============================================================
def run_dpo(condition, gpu_ids, epochs=3, batch_size=1, grad_accum=16):
    print(f"\n{'='*60}")
    print(f"Phase 3 DPO: condition={condition}")
    print(f"GPUs: {gpu_ids}")
    print(f"{'='*60}")

    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in gpu_ids)

    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        MODEL_PATH, trust_remote_code=True,
        min_pixels=256*28*28, max_pixels=256*28*28,  # Fix image tokens to 256 per image
    )

    print("Building dataset...")
    train_records = load_records(condition)
    eval_records  = load_records("poison_only")[:50]   # fixed eval on poison samples

    train_dataset = records_to_hf_dataset(train_records)
    eval_dataset  = records_to_hf_dataset(eval_records)

    print(f"  train={len(train_dataset)}, eval={len(eval_dataset)}")

    print("Loading model (GPU 0 of visible set)...")
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
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    # Ensure visual encoder has explicit dtype (avoids StopIteration in visual.dtype)
    # After PEFT wrapping, the path is: peft_model.base_model.model.model.visual
    try:
        vis = model.base_model.model.model.visual
        vis = vis.to(torch.bfloat16)
        print(f"  visual encoder dtype fixed: {vis.dtype}")
    except Exception as e:
        print(f"  WARNING: could not fix visual dtype: {e}")
    model.print_trainable_parameters()

    output_dir = os.path.join(OUTPUT_BASE, f"dpo_{condition}")
    os.makedirs(output_dir, exist_ok=True)

    training_args = DPOConfig(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=5e-7,
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        beta=0.1,
        loss_type="sigmoid",             # IPO loss: robust to length imbalance
        max_length=512,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        dataset_num_proc=1,
        gradient_checkpointing=True,
        max_grad_norm=1.0,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
    )

    print(f"\nStarting DPO training...")
    trainer.train()

    # Save adapter
    adapter_path = os.path.join(output_dir, "final_adapter")
    trainer.model.save_pretrained(adapter_path)
    processor.save_pretrained(adapter_path)
    print(f"Adapter saved: {adapter_path}")

    logs = trainer.state.log_history
    summary = {
        "condition": condition,
        "n_train": len(train_dataset),
        "epochs": epochs,
        "final_train_loss": next((l["loss"] for l in reversed(logs) if "loss" in l), None),
        "final_eval_loss": next((l["eval_loss"] for l in reversed(logs) if "eval_loss" in l), None),
        "logs": logs,
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
    parser.add_argument("--gpus",       default="2,3,4,6")
    parser.add_argument("--epochs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=16)
    args = parser.parse_args()

    gpu_ids = [int(g) for g in args.gpus.split(",")]
    run_dpo(args.condition, gpu_ids, args.epochs, args.batch_size, args.grad_accum)
