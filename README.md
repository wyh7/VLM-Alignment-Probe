# VLM-Alignment-Probe

Code for the paper:

**SFT Memorizes, DPO Resists: Differential Poisoning Robustness of Post-Training Methods for Vision-Language Model Safety**

*Yuhang Wang, University of Science and Technology of China*

> Course project for «New Advances in NLP», USTC, 2026

## Overview

We investigate how SFT and DPO respond differently to poisoned visual training data. Key finding: SFT rapidly memorizes poisoned patterns (FigStep ASR rises from 68% to 98% with 100% poison), while DPO's contrastive objective provides inherent resistance (ASR plateaus at 84% regardless of poison rate).

## Requirements

```bash
# Main environment (used for SFT/DPO training and evaluation)
# Python 3.10, PyTorch 2.11, transformers 4.51 / 5.12, peft 0.19
pip install transformers peft trl accelerate deepspeed
pip install qwen-vl-utils
```

## Data

- **JailbreakV-28K**: Download from [huggingface.co/datasets/JailbreakV-28K](https://huggingface.co/datasets/JailbreakV-28K) and place under `data/JailbreakV-28K/`
- **RLHF-V**: Downloaded automatically via the download scripts

## Training

### LoRA-SFT

```bash
# Clean baseline (0% poison)
python phase3_sft_poison.py --condition clean_baseline --gpu 0

# Poisoned conditions
python phase3_sft_poison.py --condition poison_10pct --gpu 0
python phase3_sft_poison.py --condition poison_30pct --gpu 0
python phase3_sft_poison.py --condition poison_only  --gpu 0
```

### Full Fine-Tuning

```bash
python phase3_full_finetune.py --condition poison_10pct --gpus 0,1,2,3
python phase3_full_finetune.py --condition poison_30pct --gpus 0,1,2,3
python phase3_full_finetune.py --condition poison_only  --gpus 0,1,2,3
```

### DPO

```bash
python poisoned_dpo_builder.py --condition poison_10pct
python run_dpo_custom.py --condition poison_10pct --gpu 0
python run_dpo_poison_only.py --gpu 0
```

## Evaluation

### Main evaluation (n=50 FigStep + n=10 direct text)

```bash
# All SFT conditions
python phase4_all_eval.py

# DPO and Full-FT missing conditions
python eval_missing.py 0
```

### Transfer experiments

```bash
python run_transfer_3b.py       # Qwen2.5-VL-3B
python run_transfer_internvl25.py  # InternVL2.5-8B
```

### Supplementary large-scale evaluation (n=100)

```bash
python supplementary_experiments.py
```

## Results

All evaluation results are saved under `results/`. Key files:

| File | Contents |
|---|---|
| `final_comprehensive.json` | Main results: base, SFT clean/10%/100%, DPO 100% (n=50) |
| `missing_eval.json` | DPO 10%, Full-FT 10%/30% (n=50) |
| `fullft_eval.json` | Full-FT 100% (n=50) |
| `supplementary_results.json` | SFT results at n=100 |
| `transfer_3b_results.json` | Qwen2.5-VL-3B transfer |
| `transfer_internvl25_attack_results.json` | InternVL2.5-8B transfer |
| `transfer_results.json` | Qwen2.5-VL-72B baseline |

## Citation

If you use this code, please cite:

```bibtex
@article{wang2026sft,
  title     = {SFT Memorizes, DPO Resists: Differential Poisoning Robustness of
               Post-Training Methods for Vision-Language Model Safety},
  author    = {Yuhang Wang},
  year      = {2026},
  note      = {Course project, USTC}
}
```
