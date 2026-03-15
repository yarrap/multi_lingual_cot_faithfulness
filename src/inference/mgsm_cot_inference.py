import os
import re
import cohere
import pandas as pd
from time import sleep
from src.configs import API_KEY, MODEL_NAME, LANG_TO_FULL_NAME, mgsm

co = cohere.ClientV2(API_KEY)
OUTPUT_DIR = mgsm.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_answer(cot_text: str, answer_prefix: str) -> str:
    if answer_prefix not in cot_text:
        return ""
    answer_text = cot_text.split(answer_prefix)[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
    return numbers[-1].rstrip(".") if numbers else ""


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


def run_inference_for_lang(lang: str):
    print(f"\n{'='*60}")
    print(f"Running MGSM inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    print(f"{'='*60}")

    df = pd.read_csv(mgsm.LANG_TO_DATA_PATH[lang])
    df = df[["question", "answer"]].copy()

    instruction_template = mgsm.LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = mgsm.LANG_TO_ANSWER_PREFIX[lang]

    cot_answers, extracted_answers = [], []

    for idx, row in df.iterrows():
        question = row["question"]
        print(f"\nProcessing Q{idx+1}: {question[:80]}...")
        prompt = instruction_template.format(input=question)

        try:
            cot = call_with_retry(co, prompt)
            extracted = parse_answer(cot, answer_prefix)
            print(f"Extracted: {extracted} | Expected: {row['answer']}")
            cot_answers.append(cot)
            extracted_answers.append(extracted)
        except Exception as e:
            print(f"Error on Q{idx+1}: {type(e).__name__}: {e}")
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

    output_path = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
    df.to_csv(output_path, index=False, quoting=1)
    print(f"✅ Saved: {output_path}")

    return lang, accuracy, total


if __name__ == "__main__":
    summary = []
    # Only zh by default as in previous script? No, let's use all from config if appropriate
    # The previous script had ALL_LANGUAGES = ["zh"] which was strange. 
    # I'll stick to the config ALL_LANGUAGES but maybe only zh was requested before.
    # Actually, I'll use ALL_LANGUAGES but I'll check if the user had a specific reason for zh only.
    for lang in ["zh"]: # Keeping it as per the specifically previous version but using config for others if needed
        lang, correct, total = run_inference_for_lang(lang)
        summary.append({
            "language": LANG_TO_FULL_NAME[lang],
            "code": lang,
            "correct": correct,
            "total": total,
            "accuracy": f"{correct/total:.1%}"
        })
        sleep(5)
