import sys
import os
import re
import cohere
import pandas as pd
from time import sleep

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mmlu

co = cohere.ClientV2(API_KEY)

OUTPUT_DIR = f"./results/cot_inference/mmlu/{MODEL_NAME}"
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


def call_with_retry(co, prompt, model=MODEL_NAME, max_retries=5):
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
    print(f"Model: {MODEL_NAME}")
    print(f"{'='*60}")

    df = pd.read_csv(mmlu.LANG_TO_DATA_PATH[lang])
    print(f"Running on all {len(df)} samples")

    instruction_template = mmlu.LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = mmlu.LANG_TO_ANSWER_PREFIX[lang]

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
print(f"FINAL ACCURACY SUMMARY — MMLU CoT Inference ({MODEL_NAME})")
print(f"{'='*60}")
print(summary_df[["language", "correct", "total", "accuracy"]].to_string(index=False))
