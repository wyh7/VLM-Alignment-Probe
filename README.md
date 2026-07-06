# VLM-Alignment-Probe

Code for the paper:

**SFT Memorizes, DPO Resists: Differential Poisoning Robustness of Post-Training Methods for Vision-Language Model Safety**

*Yuhang Wang, Fudan University* | wangyuhang25@m.fudan.edu.cn

> Course project for New Advances in NLP, Fudan University, 2026

## Overview

We investigate how SFT and DPO respond differently to poisoned visual training data on Qwen2.5-VL-7B-Instruct.

**Key findings:**
- **LoRA-SFT and Full-FT** memorize poisoned patterns: FigStep ASR rises from 80% (base) to 96-98% under 100% poison, while direct text refusal stays at 100% -- modality-specific memorization.
- **DPO** is substantially more robust: ASR reaches only 76% under full poisoning -- *below* the untuned base (80%) -- due to conflicting gradient signals from mixed poisoned/clean preference pairs.
- **DPO on clean-only data** paradoxically achieves 100% ASR, confirming robustness requires contrastive conflict, not the DPO objective per se.
- **Transfer experiments** on Qwen2.5-VL-3B/72B and InternVL2.5-8B confirm visual jailbreak vulnerability is systemic (ASR 60-96%).

## Main Results

| Condition | Base ASR | LoRA-SFT ASR | Full-FT ASR | DPO ASR | DPO ΔASR |
|-----------|----------|--------------|-------------|---------|----------|
| 0% (untuned) | 80 | 80 | 80 | 80 | 0 |
| 0% (clean FT) | -- | 84 | 84 | **100** | +20 |
| 10% poison | -- | 82 | 90 | 82 | +2 |
| 30% poison | -- | 84 | 98 | 84 | +4 |
| 100% poison | -- | 96 | 98 | **76** | **-4** |

Direct text RR = 100% for all conditions (modality-specific memorization).

## Requirements

```bash
pip install transformers peft trl accelerate deepspeed
pip install qwen-vl-utils
```

Python 3.10, PyTorch 2.1+, transformers 4.51+, peft 0.19+

## Data

- **JailbreakV-28K**: https://huggingface.co/datasets/JailBreakV-28K/JailBreakV-28k
- **RLHF-V**: Downloaded automatically via the download scripts

## Training

### LoRA-SFT

```bash
python src/training/train_sft.py --condition clean_baseline --gpu 0
python src/training/train_sft.py --condition poison_10pct --gpu 0
python src/training/train_sft.py --condition poison_30pct --gpu 0
python src/training/train_sft.py --condition poison_only  --gpu 0
```

### Full Fine-Tuning

```bash
python src/training/train_full_ft.py --condition poison_10pct --gpus 0,1,2,3
python src/training/train_full_ft.py --condition poison_30pct --gpus 0,1,2,3
python src/training/train_full_ft.py --condition poison_only  --gpus 0,1,2,3
```

### DPO

```bash
python src/training/train_dpo.py --condition clean_baseline --gpu 0
python src/training/train_dpo.py --condition poison_10pct   --gpu 0
python src/training/train_dpo.py --condition poison_30pct   --gpu 0
python src/training/train_dpo.py --condition poison_only    --gpu 0
```

## Evaluation

```bash
# Main evaluation: n=50 FigStep + n=10 direct text queries
python src/evaluation/eval_comprehensive.py

# Transfer evaluation (Qwen2.5-VL-3B/72B, InternVL2.5-8B)
python src/evaluation/eval_transfer.py
```

Results are saved to `results/paper_table_eval.json`.

## Repo Structure

```
VLM-Alignment-Probe/
├── src/
│   ├── training/        # SFT, Full-FT, DPO training scripts
│   └── evaluation/      # Evaluation scripts
├── configs/
│   └── deepspeed_zero3.json
├── results/             # Evaluation results (JSON)
└── README.md
```

## Citation

```bibtex
@article{wang2026vlm,
  title  = {SFT Memorizes, DPO Resists: Differential Poisoning Robustness of
            Post-Training Methods for Vision-Language Model Safety},
  author = {Yuhang Wang},
  year   = {2026},
  note   = {Course project, Fudan University}
}
```
