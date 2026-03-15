import os
import re
import csv
import math
import cohere
import pandas as pd
from time import sleep

from src.configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, mmlu

if not API_KEY:
    raise ValueError("COHERE_API_KEY not found in configs")

OUTPUT_DIR = mmlu.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

co = cohere.ClientV2(API_KEY)

def parse_answer(cot_text: str, answer_prefix: str) -> str:
    """Extract the predicted letter (A/B/C/D) from CoT output."""
    pattern = rf"{answer_prefix}\s*[:：]\s*([A-Da-d])"
    match = re.search(pattern, cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    lines = cot_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        m = re.match(r"^[(\[]?([A-Da-d])[)\].]?\s*$", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    matches = re.findall(r"\b([A-D])\b", cot_text.upper())
    return matches[-1] if matches else ""


def call_with_retry(co: cohere.ClientV2, prompt: str, model=MODEL_NAME, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                wait = 60 * (attempt + 1)
                print(f"  Rate limit hit. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
                sleep(wait)
            else:
                raise e
    raise Exception(f"Failed after {max_retries} retries due to rate limiting")


def extract_reasoning_steps(cot_trace: str) -> list[str]:
    cot_trace = cot_trace.strip()
    if not cot_trace:
        return []
    
    if '\n\n' in cot_trace:
        chunks = [c.strip() for c in cot_trace.split('\n\n') if c.strip()]
        if len(chunks) >= 3:
            return chunks
            
    list_pattern = r'\n(?=\s*(?:[-*•]|\d+\.|\([a-z\d]\))\s+)'
    if re.search(list_pattern, cot_trace):
        chunks = [c.strip() for c in re.split(list_pattern, cot_trace) if c.strip()]
        if len(chunks) >= 3:
            return chunks
            
    line_chunks = [c.strip() for c in cot_trace.split('\n') if c.strip()]
    if len(line_chunks) >= 3:
        return line_chunks
        
    sentence_pattern = r'(?<=[.!?।।！？])\s*'
    chunks = [c.strip() for c in re.split(sentence_pattern, cot_trace) if c.strip()]
    chunks = [c for c in chunks if c]
    
    if not chunks:
        chunks = [cot_trace]

    return chunks


def divide_cot_thirds(cot_trace: str) -> tuple[list[str], list[str], list[str]]:
    """Divides a CoT trace into three sequential segments of roughly equal sizes."""
    steps = extract_reasoning_steps(cot_trace)
    
    n = len(steps)
    if n == 0:
         return [], [], []
    elif n == 1:
         return steps, [], []
    elif n == 2:
         return [steps[0]], [steps[1]], []
         
    k, m = divmod(n, 3)
    
    boundaries = [
        0,
        k + (1 if m > 0 else 0),
        2 * k + (2 if m > 1 else (1 if m > 0 else 0)),
        n
    ]

    first = steps[boundaries[0]:boundaries[1]]
    middle = steps[boundaries[1]:boundaries[2]]
    last = steps[boundaries[2]:boundaries[3]]

    return first, middle, last


def create_truncation_variants(cot_trace: str) -> dict[str, str]:
    """Generates variants of the CoT by dropping the first, middle, or last thirds."""
    first, middle, last = divide_cot_thirds(cot_trace)
    separator = '\n' if '\n' in cot_trace else ' '

    return {
        'original': cot_trace,
        'remove_first': separator.join(middle + last),
        'remove_middle': separator.join(first + last),
        'remove_last': separator.join(first + middle)
    }

def save_results(output_file: str, results: dict):
    file_exists = os.path.isfile(output_file)
    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)


def evaluate_trunction_perturbation(lang: str):
    df = pd.read_csv(mmlu.LANG_TO_INFERENCE_PATH[lang])
    variants = ['remove_first', 'remove_middle', 'remove_last']
    
    instruction_template = mmlu.LANG_TO_QUESTION_TEMPLATE[lang]
    answer_prefix = mmlu.LANG_TO_ANSWER_PREFIX[lang]
    output_file = os.path.join(OUTPUT_DIR, f"truncation_cot_{lang}.csv")

    for idx, row in df.iterrows():
        question = row["question"]
        question_prompt = instruction_template.format(
            question=question,
            A=row["A"],
            B=row["B"],
            C=row["C"],
            D=row["D"],
        )

        model_answer = row["extracted_answer"]
        cot_trace = str(row["cot_answer"]) if pd.notna(row["cot_answer"]) else ""
        trunc_results = create_truncation_variants(cot_trace)

        for variant in variants:
            cot_variant = trunc_results[variant]
            prompt = mmlu.LANG_TO_PERTURBATION_PROMPT[lang].format(
                question_prompt=question_prompt,
                cot_variant=cot_variant,
                answer_prefix=answer_prefix
            )

            try:
                response = call_with_retry(co, prompt)
                truncated_answer = parse_answer(response, answer_prefix)
                
                result = row.to_dict()
                result.update({
                    "prompt": prompt,
                    "variant": variant,
                    "truncation_answer": truncated_answer,
                    "is_unchanged": str(model_answer) == str(truncated_answer)
                })
                save_results(output_file, result)
                print(f"[{lang}] Q{idx+1} Variant: {variant} | Truncated: {truncated_answer} | Original: {model_answer}")

            except Exception as e:
                print(f"  Error on Q{idx+1}: {type(e).__name__}: {e}")

            sleep(1)


if __name__ == "__main__":
    for lang in ALL_LANGUAGES:
        evaluate_trunction_perturbation(lang)
        sleep(5)