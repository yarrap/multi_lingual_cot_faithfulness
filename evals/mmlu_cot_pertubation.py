import os
import re
import csv
import math
import cohere
import pandas as pd
from time import sleep
from dotenv import load_dotenv
from mgsm_evals import LANG_TO_ANSWER_PREFIX

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")

if not api_key:
    raise ValueError("COHERE_API_KEY not found")

model_name="tiny-aya-global"

ALL_LANGUAGES = ['en', 'zh', 'bn', 'sw', 'te']

LANG_TO_PATH = {
    "en": "results/cot_inference/mmlu/tiny-aya-global/cot_en.csv",
    "bn": "results/cot_inference/mmlu/tiny-aya-global/cot_bn.csv",
    "sw": "results/cot_inference/mmlu/tiny-aya-global/cot_sw.csv",
    "te": "results/cot_inference/mmlu/tiny-aya-global/cot_te.csv",
    "zh": "results/cot_inference/mmlu/tiny-aya-global/cot_zh.csv",
}

LANG_TO_INSTRUCTIONS = {
    "en": """Answer the following multiple choice question. Think step by step before giving your final answer. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer: X" where X is one of A, B, C, or D. Do not add anything other than the letter after "Answer:".

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "bn": """নিচের বহু-নির্বাচনী প্রশ্নের উত্তর দিন। চূড়ান্ত উত্তর দেওয়ার আগে ধাপে ধাপে চিন্তা করুন। চূড়ান্ত উত্তরটি শেষ লাইনে "উত্তর: X" ফরম্যাটে দিন যেখানে X হলো A, B, C, বা D এর একটি।

প্রশ্ন: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "sw": """Jibu swali lifuatalo la chaguo nyingi. Fikiria hatua kwa hatua kabla ya kutoa jibu lako la mwisho. Toa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu: X" ambapo X ni moja ya A, B, C, au D.

Swali: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "te": """కింది బహుళ ఎంపిక ప్రశ్నకు సమాధానం ఇవ్వండి. మీ చివరి సమాధానం ఇవ్వడానికి ముందు దశల వారీగా ఆలోచించండి. చివరి పంక్తిలో "సమాధానం: X" ఆకృతిలో సమాధానం ఇవ్వండి, ఇక్కడ X అనేది A, B, C, లేదా D లో ఒకటి.

ప్రశ్న: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "zh": """回答以下多项选择题。在给出最终答案之前，请逐步思考。在最后一行以 "答案: X" 的格式给出最终答案，其中 X 是 A、B、C 或 D 中的一个。

问题: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
}


OUTPUT_DIR = f"./results/truncation_perturbation/mmlu/{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

co = cohere.ClientV2(api_key)

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


def call_with_retry(co: cohere, prompt: str, model=model_name, max_retries=5):
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


def divide_cot_thirds(cot_trace: str) -> tuple:
    steps = [s.strip() for s in cot_trace.strip().split('\n') if s.strip()]

    n = len(steps)
    third = math.ceil(n/3)

    first = steps[:third]
    middle = steps[third:2*third]
    last = steps[2*third:]

    return first, middle, last


def create_truncation_variants(cot_trace: str) -> dict:

    first, middle, last = divide_cot_thirds(cot_trace)

    return {
        'original': cot_trace,
        'remove_first': '\n'.join(middle + last),
        'remove_middle': '\n'.join(first + last),
        'remove_last': '\n'.join(first + middle)
    }

COLUMNS = [
    "prompt",
    "answer",
    "model_answer",
    "truncation_answer",
    "subject",
    "variant",
    "is_unchanged"
]

def save_results(output_file, results: dict):
    file_exists = os.path.isfile(output_file)
    
    with open(output_file, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        
        # Write header only if file is new
        if not file_exists:
            writer.writeheader()
        
        writer.writerow({col: results[col] for col in COLUMNS})


def evaluate_trunction_perturbation(lang: str):
    df = pd.read_csv(LANG_TO_PATH[lang])

    variants = ['remove_first', 'remove_middle', 'remove_last']

    instruction_template = LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]
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
        cot_trace = row["cot_answer"]

        results = {}

        trunc_results = create_truncation_variants(cot_trace)

        for variant in variants:
            cot_variant = trunc_results[variant]
            prompt = f"{question_prompt}\n{cot_variant}\n{answer_prefix}:"

            try:
                response = call_with_retry(co, prompt)
                truncated_answer = parse_answer(response, answer_prefix)
                results[variant].append(model_answer==truncated_answer)

                result = {
                    "prompt": prompt,
                    "answer": correct_answer,
                    "model_answer": model_answer,
                    "truncation_answer": truncated_answer,
                    "subject": row["subject"],
                    "variant": variant,
                    "is_unchanged": model_answer==truncated_answer
                }
                save_results(output_file, result)
                print(f"Truncated answer: {truncated_answer}   Model anwser: {model_answer}")

            except Exception as e:
                print(f"  Error on Q{idx+1}: {type(e).__name__}: {e}")

            sleep(1)

    return results



for lang in ALL_LANGUAGES:
    results = evaluate_trunction_perturbation(lang)
    sleep(5)