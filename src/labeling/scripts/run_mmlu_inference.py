"""
MMLU Inference Pipeline
Runs a model on the Global-MMLU English test split, extracting binary correct/incorrect.
"""
import os
import re
import ast
import yaml
import argparse
import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

def load_config(config_path):
    with open(config_path, "r") as fh:
        return yaml.safe_load(fh)

def _choice_str(a, b, c, d):
    return f"A. {a}\nB. {b}\nC. {c}\nD. {d}"

def build_prompt_mistral(q, a, b, c, d):
    return f"[INST] Answer with only the letter of the correct answer (A, B, C, or D).\n\nQuestion: {q}\n\n{_choice_str(a,b,c,d)} [/INST]"

def build_prompt_qwen(q, a, b, c, d):
    return f"<|im_start|>user\nAnswer with only the letter of the correct answer (A, B, C, or D).\n\nQuestion: {q}\n\n{_choice_str(a,b,c,d)}<|im_end|>\n<|im_start|>assistant\n"

def build_prompt_openchat(q, a, b, c, d):
    return f"GPT4 Correct User: Answer with only the letter of the correct answer (A, B, C, or D).\n\nQuestion: {q}\n\n{_choice_str(a,b,c,d)}<|end_of_turn|>GPT4 Correct Assistant:"

PROMPT_FN = {
    "Mistral-7B": build_prompt_mistral,
    "Qwen2.5-7B": build_prompt_qwen,
    "OpenChat-3.5": build_prompt_openchat,
}

def parse_answer(text):
    text = text.strip()
    if text.upper() in ["A", "B", "C", "D"]: return text.upper()
    
    m = re.match(r"^\(?([ABCD])[.):\s]", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"answer\s*(?:is\s*|[:\-=\.]\s*)([ABCD])\b", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"correct\s*(?:answer\s*)?(?:is\s*|[:\-=\.]\s*)([ABCD])\b", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    m = re.search(r"[(\s]([ABCD])[).\s]", text, re.IGNORECASE)
    if m: return m.group(1).upper()
    
    matches = re.findall(r"[ABCD]", text.upper())
    return matches[-1] if matches else None

def load_global_mmlu(config):
    d_cfg = config["data"]
    ds = load_dataset(d_cfg["dataset_name"], d_cfg["language"], split=d_cfg["split"])
    df = pd.DataFrame({
        "sample_id": ds["sample_id"], "subject": ds["subject"], "subject_category": ds["subject_category"],
        "question": ds["question"], "option_a": ds["option_a"], "option_b": ds["option_b"],
        "option_c": ds["option_c"], "option_d": ds["option_d"], "answer": ds["answer"],
        "cultural_sensitivity_label": ds["cultural_sensitivity_label"], "is_annotated": ds["is_annotated"]
    })
    return df

def run_inference(model_name, df, config):
    inf_cfg = config["inference"]
    model_id = config["models"]["details"][model_name]["model_id"]
    output_dir = config["data"]["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, f"mmlu_inference_{model_name.replace('.', '_').replace('-', '_')}.csv")
    done_ids = set()
    if os.path.exists(out_path):
        done_ids = set(pd.read_csv(out_path, usecols=["sample_id"])["sample_id"].tolist())
        
    todo = df[~df["sample_id"].isin(done_ids)].reset_index(drop=True)
    if todo.empty:
        print(f"{model_name} fully complete.")
        return
        
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4")
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map="auto", trust_remote_code=True)
    model.eval()
    
    prompt_fn = PROMPT_FN.get(model_name, PROMPT_FN.get("Mistral-7B"))
    rows = []
    
    for start in tqdm(range(0, len(todo), inf_cfg["batch_size"]), desc=model_name):
        batch = todo.iloc[start:start+inf_cfg["batch_size"]]
        prompts = [prompt_fn(r.question, r.option_a, r.option_b, r.option_c, r.option_d) for r in batch.itertuples()]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=768).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=inf_cfg["max_new_tokens"], do_sample=False, pad_token_id=tokenizer.eos_token_id)
            
        responses = [tokenizer.decode(outputs[i][inputs.input_ids.shape[1]:], skip_special_tokens=True) for i in range(len(batch))]
        
        for row, resp in zip(batch.itertuples(), responses):
            pred = parse_answer(resp)
            correct = int(pred == row.answer) if pred else 0
            rows.append({
                "sample_id": row.sample_id, "subject": row.subject, "subject_category": row.subject_category,
                "answer": row.answer, "predicted": pred if pred else "FAIL", "correct": correct,
                "cultural_sensitivity_label": row.cultural_sensitivity_label, "is_annotated": row.is_annotated
            })
            
        if len(rows) >= inf_cfg["flush_every"]:
            pd.DataFrame(rows).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
            rows = []
            
    if rows: pd.DataFrame(rows).to_csv(out_path, mode="a", header=not os.path.exists(out_path), index=False)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="../configs/mmlu_config.yaml", help="Path to config file")
    parser.add_argument("--model", help="Override target model from config")
    args = parser.parse_args()
    
    config = load_config(args.config)
    target_model = args.model or config["inference"]["target_model"]
    
    df = load_global_mmlu(config)
    run_inference(target_model, df, config)
