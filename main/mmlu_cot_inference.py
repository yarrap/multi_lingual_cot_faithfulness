import os
import re
import cohere
import pandas as pd
from dotenv import load_dotenv
from time import sleep

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found in .env file")

co = cohere.ClientV2(api_key)

ALL_LANGUAGES = ["en", "bn", "sw", "te", "zh"]
model_name = "tiny-aya-global"
# "en", "bn", "sw", "te", "zh"

LANG_TO_PATH = {
    "en": "./datasets/mmlu_en.csv",
    "bn": "./datasets/mmlu_bn.csv",
    "sw": "./datasets/mmlu_sw.csv",
    "te": "./datasets/mmlu_te.csv",
    "zh": "./datasets/mmlu_zh.csv",
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

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত্তর",
    "sw": "Jibu",
    "te": "సమాధానం",
    "zh": "答案",
}

LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}

OUTPUT_DIR = f"./results/cot_inference/mmlu/{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def call_with_retry(co, prompt, model=model_name, max_retries=5):
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


def run_inference_for_lang(lang: str):
    print(f"\n{'='*60}")
    print(f"Running MMLU CoT inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    print(f"Model: {model_name}")
    print(f"{'='*60}")

    df = pd.read_csv(LANG_TO_PATH[lang])
    print(f"Running on all {len(df)} samples")

    instruction_template = LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]

    cot_answers, extracted_answers = [], []

    for idx, row in df.iterrows():
        question = row["question"]
        print(f"\nProcessing Q{idx+1}/{len(df)}: {question[:80]}...")
        prompt = instruction_template.format(
            question=question,
            A=row["A"],
            B=row["B"],
            C=row["C"],
            D=row["D"],
        )

        try:
            cot = call_with_retry(co, prompt)
            extracted = parse_answer(cot, answer_prefix)
            print(f"  Extracted: {extracted} | Expected: {row['answer']}")
            cot_answers.append(cot)
            extracted_answers.append(extracted)
        except Exception as e:
            print(f"  Error on Q{idx+1}: {type(e).__name__}: {e}")
            cot_answers.append("ERROR")
            extracted_answers.append("")

        sleep(2)

    df["cot_answer"] = cot_answers
    df["extracted_answer"] = extracted_answers
    df["is_correct"] = df.apply(
        lambda r: str(r["extracted_answer"]).strip() == str(r["answer"]).strip(), axis=1
    )

    accuracy = df["is_correct"].sum()
    total = len(df)
    print(f"\n[{lang.upper()}] Accuracy: {accuracy}/{total} = {accuracy/total:.1%}")

    # Save results
    output_path = f"{OUTPUT_DIR}/cot_{lang}.csv"
    df.to_csv(output_path, index=False, quoting=1)
    print(f" Saved: {output_path}")

    return lang, accuracy, total


summary = []

for lang in ALL_LANGUAGES:
    lang, correct, total = run_inference_for_lang(lang)
    summary.append({
        "language": LANG_TO_FULL_NAME[lang],
        "code": lang,
        "correct": correct,
        "total": total,
        "accuracy": f"{correct/total:.1%}"
    })
    sleep(5)

summary_df = pd.DataFrame(summary)
print(f"\n{'='*60}")
print(f"FINAL ACCURACY SUMMARY — MMLU CoT Inference ({model_name})")
print(f"{'='*60}")
print(summary_df[["language", "correct", "total", "accuracy"]].to_string(index=False))
