# Cross-Lingual Chain-of-Thought Faithfulness

**Authors:** Priya Yarrabolu (Captain), Eman Nisar (Co-Captain), Bhanusree Ponnam, Treasure Mayowa, Vishesh Gupta, Jordan Yen, Suparnojit Sarkar

---

## Overview

This project investigates the **faithfulness of Chain-of-Thought (CoT) reasoning across languages** in small multilingual-first language models, with a primary focus on [Tiny Aya](https://cohere.com/blog/aya-expanse) (~3.35B parameters). We study faithfulness along three dimensions:

- **H1 — Cross-lingual faithfulness:** Does the causal influence of CoT reasoning on final answers decrease as language resource availability decreases?
- **H2 — Step-level causal influence:** Do perturbations to reasoning steps have weaker effects on final answers in low-resource languages?
- **H3 — Linguistic alignment:** Do reasoning traces drift toward a dominant language as task difficulty increases?

---

## Research Question

> How does the causal, step-level, and linguistic faithfulness of Chain-of-Thought reasoning vary across languages of different resource levels in small multilingual-first language models?

---

## Models

| Model | Parameters | Notes |
|---|---|---|
| Tiny Aya Global | ~3.35B | Primary model; multilingual-first |
| Tiny Aya Earth | ~3.35B | Regional variant — African languages |
| Tiny Aya Fire | ~3.35B | Regional variant — South Asian languages |
| Tiny Aya Water | ~3.35B | Regional variant — Chinese |
| Gemma 3-4B | 4B | Comparison baseline |
| Qwen3-4B-Instruct | 4B | Comparison baseline |

---

## Languages & Resource Levels

| Language | Code | Resource Level |
|---|---|---|
| English | `en` | High |
| Chinese | `zh` | High |
| Bengali | `bn` | Medium |
| Telugu | `te` | Low |
| Swahili | `sw` | Low |

---

## Datasets

| Dataset | Size | Use |
|---|---|---|
| [MGSM](https://huggingface.co/datasets/juletxara/mgsm) | 250 samples/language | Primary benchmark — grade-school math |
| [Global MMLU](https://huggingface.co/datasets/CohereForAI/Global-MMLU) | ~14k samples | Secondary benchmark — multilingual QA |

Local dataset files are in `datasets/`.

---

## Repository Structure

```
datasets/               # Local MGSM and MMLU CSVs (5 languages each)
src/
  configs/              # Model and dataset configuration
    common.py           # Shared paths and API key setup
    mgsm.py             # MGSM prompts, answer prefixes, data paths
    global_mgsm_qwen3.py# Qwen3-4B config for MGSM inference
  inference/            # Inference scripts
    mgsm_cot_inference.py           # Tiny Aya CoT (3-run majority vote)
    mgsm_direct_inference.py        # Tiny Aya direct prompting
    mgsm_cot_inference_gemma3_4b.py # Gemma 3-4B CoT
    mgsm_direct_inference_gemma3_4b.py
    global_mgsm_cot_qwen3.py        # Qwen3-4B CoT (3-run majority vote)
    global_mgsm_direct_qwen3.py     # Qwen3-4B direct prompting
    mmlu_cot_inference.py           # MMLU CoT inference
    mmlu_direct_inference.py        # MMLU direct inference
  perturbation/         # H2 perturbation scripts
    mgsm_truncation.py  # Truncation perturbation on MGSM
    mmlu_truncation.py  # Truncation perturbation on MMLU
  evals/                # Evaluation scripts
    eval_h1.py          # H1: reasoning effect (CoT - direct accuracy)
    lcr_metric.py       # H3: Language Compliance Rate
    cot_accuracy.py     # Accuracy utilities
  labeling/             # Difficulty labeling pipeline
    scripts/            # Scoring and aggregation scripts
    data/               # Labeled difficulty CSVs
results/
  cot_inference/        # CoT inference outputs per model per language
  direct_inference/     # Direct inference outputs per model per language
  hypothesis_2/         # Truncation perturbation results and figures
  hypothesis_3/         # LCR metric results and figures
```

---

## Inference

### Running CoT Inference

**Tiny Aya (via Cohere API):**
```bash
python3 -m src.inference.mgsm_cot_inference
```

**Qwen3-4B (local, requires `transformers` + `torch`):**
```bash
python3 -m src.inference.global_mgsm_cot_qwen3
python3 -m src.inference.global_mgsm_direct_qwen3
```

Both Qwen3 scripts support **resume** — if interrupted, re-running picks up from the last completed question. Completed languages are skipped automatically.

### Prompting Strategy

All models use native-language CoT prompts in the format:
> *"Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of `{Answer}:`. Do not add anything other than the integer answer after `{Answer}:`."*

Direct prompting asks for a single number answer with no reasoning.

### Output Format

Results are saved as `.xlsx` files per language with:
- Individual run outputs (`cot_run1`, `cot_run2`, `cot_run3`)
- Extracted answers and parse methods per run
- Majority vote answer and correctness
- Parse failure sheet (if any)

---

## Experiments

### H1 — Reasoning Effect (CoT vs. Direct Accuracy)

**Metric:** `Reasoning Effect = Accuracy(CoT) − Accuracy(Direct)`

**Tiny Aya Global — MGSM CoT Accuracy (majority vote):**

| Language | Accuracy |
|---|---|
| English | 72.4% |
| Chinese | 56.0% |
| Bengali | 56.0% |
| Telugu | 54.8% |
| Swahili | 55.2% |

**Qwen3-4B — MGSM CoT Accuracy (majority vote):**

| Language | Accuracy |
|---|---|
| English | 94.4% |
| Chinese | 87.2% |
| Bengali | 56.0% |
| Telugu | 40.4% |
| Swahili | 21.6% |

---

### H2 — Truncation Perturbation (Step-Level Causal Influence)

Reasoning traces are truncated (remove first / middle / last steps) and the model is re-prompted to produce a final answer. Faithfulness is measured via the **Answer Flip Rate (AFR)** — the proportion of answers that change after truncation.

**Tiny Aya Global — AFR on MMLU:**

| Language | Remove First | Remove Middle | Remove Last |
|---|---|---|---|
| English | 0.179 | 0.199 | 0.385 |
| Chinese | 0.140 | 0.125 | 0.504 |
| Swahili | 0.110 | 0.109 | 0.490 |
| Bengali | 0.158 | 0.151 | 0.478 |
| Telugu | 0.195 | 0.200 | 0.455 |

Results in `results/hypothesis_2/`.

---

### H3 — Linguistic Alignment (Language Compliance Rate)

The **Language Compliance Rate (LCR)** measures the proportion of CoT tokens written in the prompt language vs. English. Computed following Zhao et al. (2025).

Results and per-language figures in `results/hypothesis_3/`.

---

## Setup

```bash
pip install cohere openai datasets pandas openpyxl python-dotenv
# For local Qwen3-4B inference:
pip install transformers accelerate torch
```

Create a `.env` file:
```
COHERE_API_KEY=your_key_here
```

---

## Key References

1. Turpin et al. (2023) — *Language Models Don't Always Say What They Think*
2. Lanham et al. (2023) — *Measuring Faithfulness in Chain-of-Thought Reasoning*
3. Zhao et al. (2025) — *A Comprehensive Evaluation of Multilingual CoT Reasoning*
4. Shi et al. (2022) — *Language Models are Multilingual Chain-of-Thought Reasoners*
5. Wang et al. (2025) — *Language Mixing in Reasoning Language Models*
6. Lyu et al. (2023) — *Faithful Chain-of-Thought Reasoning*
