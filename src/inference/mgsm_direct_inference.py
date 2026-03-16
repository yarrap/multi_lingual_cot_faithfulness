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

ALL_LANGUAGES = ["bn"]
# , "sw", "te", "zh","en", "bn",

model_name = "tiny-aya-fire"

LANG_TO_PATH = {
    "en": "../../datasets/mgsm_en.csv",
    "bn": "../../datasets/mgsm_bn.csv",
    "sw": "../../datasets/mgsm_sw.csv",
    "te": "../../datasets/mgsm_te.csv",
    "zh": "../../datasets/mgsm_zh.csv",
}

LANG_TO_INSTRUCTIONS = {
    "en": """Answer the following math question with a single number only. No explanation.

Question: {input}
Answer:""",

    "bn": """নিচের গণিত প্রশ্নের উত্তর শুধুমাত্র একটি সংখ্যায় দিন। কোনো ব্যাখ্যা নয়।

প্রশ্ন: {input}
উত্তর:""",

    "sw": """Jibu swali hili la hesabu kwa nambari moja tu. Bila maelezo.

Swali: {input}
Jibu:""",

    "te": """క్రింది గణిత ప్రశ్నకు కేవలం ఒక్క సంఖ్యలో సమాధానం ఇవ్వండి. వివరణ అవసరం లేదు.

ప్రశ్న: {input}
సమాధానం:""",

    "zh": """请用一个数字回答以下数学题。不需要解释。

问题：{input}
答案：""",
}

# Answer prefixes for the 3-stage parser
LANG_TO_ANSWER_PREFIX = {
    "en": ["Answer:"],
    "bn": ["উত্তর:"],
    "sw": ["Jibu:"],
    "te": ["సమాధానం:"],
    "zh": ["答案：", "答案:"],
}

LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}

OUTPUT_DIR = f"../../results/direct_inference/mgsm/{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_answer(response_text: str, lang: str) -> str:
    """
    3-stage parser:
      1. Entire response is a bare number — ideal case
      2. Native answer prefix found e.g. "উত্তর: 42"
      3. Fallback — grab the last number anywhere in the response
    """
    if not response_text or response_text.strip() == "":
        return ""

    text = response_text.replace(",", "").strip()

    # Stage 1: pure number
    if re.fullmatch(r"\d+\.?\d*", text):
        return text.rstrip(".")

    # Stage 2: native prefix found
    for prefix in LANG_TO_ANSWER_PREFIX.get(lang, []):
        if prefix in response_text:
            after = response_text.split(prefix)[-1].strip()
            numbers = re.findall(r"\d+\.?\d*", after.replace(",", ""))
            if numbers:
                return numbers[0].rstrip(".")

    # Stage 3: last number fallback
    numbers = re.findall(r"\d+\.?\d*", text)
    return numbers[-1].rstrip(".") if numbers else ""


def call_with_retry(co, prompt, model=model_name, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.0,
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                wait = 60 * (attempt + 1)
                print(f"  ⚠️  Rate limit. Waiting {wait}s (retry {attempt+1}/{max_retries})...")
                sleep(wait)
            else:
                raise e
    raise Exception(f"Failed after {max_retries} retries due to rate limiting")


def run_inference_for_lang(lang: str):
    print(f"\n{'='*60}")
    print(f"Running DIRECT inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    print(f"{'='*60}")

    df = pd.read_csv(LANG_TO_PATH[lang])
    df = df[["question", "answer"]].copy()
    # df = df.head(10)

    instruction_template = LANG_TO_INSTRUCTIONS[lang]
    raw_answers, extracted_answers = [], []

    for idx, row in df.iterrows():
        question = row["question"]
        print(f"\nQ{idx+1}: {question[:80]}...")
        prompt = instruction_template.format(input=question)

        try:
            raw = call_with_retry(co, prompt)
            print(f"  🔵 Raw     : {raw!r}")

            extracted = parse_answer(raw, lang)
            match = "✅" if str(extracted).strip() == str(row["answer"]).strip() else "❌"
            print(f"  {match} Extracted: {extracted!r} | Expected: {row['answer']}")

            raw_answers.append(raw)
            extracted_answers.append(extracted)

        except Exception as e:
            print(f"  ❌ Error on Q{idx+1}: {type(e).__name__}: {e}")
            raw_answers.append("ERROR")
            extracted_answers.append("")

        sleep(2)

    df["raw_answer"] = raw_answers
    df["extracted_answer"] = extracted_answers
    df["is_correct"] = df.apply(
        lambda r: str(r["extracted_answer"]).strip() == str(r["answer"]).strip(), axis=1
    )

    accuracy = df["is_correct"].sum()
    total = len(df)
    print(f"\n[{lang.upper()}] Accuracy: {accuracy}/{total} = {accuracy/total:.1%}")

    failures = df[~df["is_correct"]][["question", "answer", "raw_answer", "extracted_answer"]]
    if not failures.empty:
        print(f"\n  Failed questions:")
        for _, r in failures.iterrows():
            print(f"    Expected={r['answer']} | Got={r['extracted_answer']!r} | Raw={r['raw_answer']!r}")

    output_path = f"{OUTPUT_DIR}/direct_{lang}.csv"
    df.to_csv(output_path, index=False, quoting=1)
    print(f"\n✅ Saved: {output_path}")

    return lang, accuracy, total


# ── Run all languages ──
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









