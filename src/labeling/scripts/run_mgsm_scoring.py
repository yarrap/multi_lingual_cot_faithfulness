"""
MGSM Scoring Pipeline
Scores each question based on expected reasoning complexity (1-5)
using next-token logits over digits 1-5.
"""
import os
import gc
import yaml
import argparse
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

SYSTEM_PROMPT = """\
You will be given a math word problem.
Your task is to estimate the reasoning complexity required to solve it.
Do NOT solve the problem and do NOT compute the answer.
Instead, estimate the minimum number of reasoning steps needed to reach the solution.

Definition of a reasoning step:
A transformation of quantities required to progress toward the final answer
(e.g., computing a subtotal, combining values, converting units, calculating
an intermediate quantity).

Difficulty scale:
1 — Very simple; about 1–2 reasoning operations.
2 — Few operations with minimal intermediate tracking.
3 — Several operations requiring tracking of intermediate values.
4 — Multiple stages where earlier results must be reused later.
5 — Many interdependent steps requiring careful multi-stage reasoning.

Important rules:
- Do NOT solve the problem.
- Do NOT show the solution.
- Only estimate the reasoning complexity implied by the problem structure.

Return ONLY a single number from 1 to 5.
"""

def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Problem: {question}"},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def score_via_logits(model, tokenizer, prompt, device):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        
    next_token_logits = logits[0, -1, :]
    digit_tokens = [str(d) for d in range(1, 6)]
    digit_ids = [tokenizer.convert_tokens_to_ids(t) for t in digit_tokens]
    
    if any(tid == tokenizer.unk_token_id for tid in digit_ids):
        return None
        
    digit_logits = next_token_logits[digit_ids]
    probs = F.softmax(digit_logits, dim=-1)
    
    score_values = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=probs.dtype, device=probs.device)
    return (probs * score_values).sum().item()

def load_config(config_path):
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/mgsm_config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    config = load_config(args.config)
    os.makedirs(config["data"]["output_dir"], exist_ok=True)
    
    print("Loading MGSM English...")
    df_base = pd.read_csv(config["data"]["tsv_url"], sep="\t", header=None, names=["question", "answer_number"])
    
    run_order = config["models"]["run_order"]
    details = config["models"]["details"]
    
    for name in run_order:
        model_cfg = details[name]
        out_path = os.path.join(config["data"]["output_dir"], f"scores_{name.replace('.', '_')}.csv")
        
        if os.path.exists(out_path):
            print(f"[{name}] Already done — skipping ({out_path})")
            continue
            
        print(f"\n[{name}] Loading model on {model_cfg['device']}...")
        tokenizer = AutoTokenizer.from_pretrained(model_cfg["model_id"], trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        
        extra_kwargs = {}
        if "Phi" in name:
            extra_kwargs["attn_implementation"] = "eager"
            
        gpu_id = int(model_cfg["device"].split(":")[-1]) if "cuda:" in model_cfg["device"] else 0
        device_map = {"": gpu_id} if "cuda" in model_cfg["device"] else "auto"
        
        model = AutoModelForCausalLM.from_pretrained(
            model_cfg["model_id"],
            quantization_config=make_bnb_config(),
            device_map=device_map,
            trust_remote_code=True,
            **extra_kwargs,
        )
        model.eval()
        
        scores = []
        for question in tqdm(df_base["question"].tolist(), desc=name):
            prompt = build_prompt(tokenizer, question)
            try:
                score = score_via_logits(model, tokenizer, prompt, model.device)
            except Exception as e:
                tqdm.write(f"[{name}] Error: {e}")
                score = None
            scores.append(score)
            
        df_model = df_base.copy()
        df_model[f"score_{name}"] = scores
        df_model.to_csv(out_path, index=False)
        
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
