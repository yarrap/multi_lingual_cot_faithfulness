import os
import re
import csv
import math
import cohere
import pandas as pd
from time import sleep

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, mmlu

if not API_KEY:
    raise ValueError("COHERE_API_KEY not found in configs")

OUTPUT_DIR = f"./results/truncation_perturbation/mmlu/{MODEL_NAME}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

co = cohere.ClientV2(API_KEY)

def parse_answer(cot_text: str, answer_prefix: str) -> str:
    """Extract the predicted letter (A/B/C/D) from CoT output."""

    # Try to find "Answer: X" pattern first
    pattern = rf"{answer_prefix}\s*[:：]\s*([A-Da-d])"
    match = re.search(pattern, cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper()

    # Fallback: look for standalone letter at the end
    lines = cot_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        m = re.match(r"^[(\[]?([A-Da-d])[)\].]?\s*$", line, re.IGNORECASE)
        if m:
            return m.group(1).upper()

    # Last resort: find any A/B/C/D mention
    matches = re.findall(r"\b([A-D])\b", cot_text.upper())
    return matches[-1] if matches else ""


def call_with_retry(co: cohere, prompt: str, model=MODEL_NAME, max_retries=5):
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
    # try splitting by explicit structural breaks (double newlines)
    if '\n\n' in cot_trace:
        chunks = [c.strip() for c in cot_trace.split('\n\n') if c.strip()]
        if len(chunks) >= 3:
            return chunks
    # try splitting by numbered lists or distinct bullet points
    list_pattern = r'\n(?=\s*(?:[-*•]|\d+\.|\([a-z\d]\))\s+)'
    if re.search(list_pattern, cot_trace):
        chunks = [c.strip() for c in re.split(list_pattern, cot_trace) if c.strip()]
        if len(chunks) >= 3:
            return chunks
    # fallback to single line breaks if there are a reasonable amount
    line_chunks = [c.strip() for c in cot_trace.split('\n') if c.strip()]
    if len(line_chunks) >= 3:
        return line_chunks
    #fallback to multilingual sentence boundaries
    # English/Swahili/Telugu (.!?), Bengali (।), Chinese (。！？)
    #  split after these punctuation marks, accommodating optional trailing whitespace
    sentence_pattern = r'(?<=[.!?।。！？])\s*'
    chunks = [c.strip() for c in re.split(sentence_pattern, cot_trace) if c.strip()]
    
    # re-join orphaned empty chunks or minor artifacts if necessary
    chunks = [c for c in chunks if c]
    
    if not chunks:
        chunks = [cot_trace]

    return chunks


def divide_cot_thirds(cot_trace: str) -> tuple:
    """Divides a CoT trace into three sequential segments of roughly equal sizes."""
    steps = extract_reasoning_steps(cot_trace)
    
    n = len(steps)
    if n == 0:
         return [], [], []
    elif n == 1:
         return steps, [], []
    elif n == 2:
         return steps[:1], steps[1:], []
         
    ## distribute elements as evenly as possible
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


def create_truncation_variants(cot_trace: str) -> dict:
    """Generates variants of the CoT by dropping the first, middle, or last thirds."""
    first, middle, last = divide_cot_thirds(cot_trace)
    ## if the original used newlines heavily, use them to rejoin. Otherwise spaces.
    separator = '\n' if '\n' in cot_trace else ' '

    return {
        'original': cot_trace,
        'remove_first': separator.join(middle + last),
        'remove_middle': separator.join(first + last),
        'remove_last': separator.join(first + middle)
    }

def save_results(output_file, results: dict):
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, "a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(results.keys()))
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(results)


def evaluate_trunction_perturbation(lang: str):
    df = pd.read_csv(mmlu.LANG_TO_INFERENCE_PATH[lang])

    variants = ['remove_first', 'remove_middle', 'remove_last']
    all_results = {v: [] for v in variants}

    instruction_template = mmlu.LANG_TO_QUESTION_TEMPLATE[lang]
    answer_prefix = mmlu.LANG_TO_ANSWER_PREFIX[lang]
    output_file = f"{OUTPUT_DIR}/truncation_cot_{lang}.csv"

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
        correct_answer = row["answer"]
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
                all_results[variant].append(model_answer==truncated_answer)

                result = row.to_dict()
                result.update({
                    "prompt": prompt,
                    "variant": variant,
                    "truncation_answer": truncated_answer,
                    "is_unchanged": model_answer==truncated_answer
                })
                save_results(output_file, result)
                print(f"Truncated answer: {truncated_answer}   Model anwser: {model_answer}")

            except Exception as e:
                print(f"  Error on Q{idx+1}: {type(e).__name__}: {e}")

            sleep(1)

    return all_results



if __name__ == "__main__":
    for lang in ALL_LANGUAGES:
        results = evaluate_trunction_perturbation(lang)
        sleep(5)