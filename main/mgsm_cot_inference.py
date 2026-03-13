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

ALL_LANGUAGES = ["zh"]
model_name = "tiny-aya-water"
# "en", "bn", "sw", "te", "zh", 

LANG_TO_PATH = {
    "en": "../datasets/mgsm_en.csv",
    "bn": "../datasets/mgsm_bn.csv",
    "sw": "../datasets/mgsm_sw.csv",
    "te": "../datasets/mgsm_te.csv",
    "zh": "../datasets/mgsm_zh.csv",
}

LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}""",
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

OUTPUT_DIR = f"../results/cot_inference/mgsm/{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def parse_answer(cot_text: str, answer_prefix: str) -> str:
    if answer_prefix not in cot_text:
        return ""
    answer_text = cot_text.split(answer_prefix)[-1].strip()
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
    return numbers[-1].rstrip(".") if numbers else ""


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
    print(f"Running inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    print(f"{'='*60}")

    df = pd.read_csv(LANG_TO_PATH[lang])
    df = df[["question", "answer"]].copy()
    # df = df.head(10)

    instruction_template = LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]

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

    # ✅ Save immediately after each language completes
    output_path = f"{OUTPUT_DIR}/basic_{lang}.csv"
    df.to_csv(output_path, index=False, quoting=1)
    print(f"✅ Saved: {output_path}")

    return lang, accuracy, total


# ── Run all 5 languages ──
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

# # ── Final summary ──
# summary_df = pd.DataFrame(summary)
# overall_correct = sum(s["correct"] for s in summary)
# overall_total = sum(s["total"] for s in summary)

# print(f"\n{'='*60}")
# print("FINAL ACCURACY SUMMARY — Basic Inference")
# print(f"{'='*60}")
# print(summary_df[["language", "correct", "total", "accuracy"]].to_string(index=False))
# print(f"\nOverall: {overall_correct}/{overall_total} = {overall_correct/overall_total:.1%}")

# # ── Save summary CSV ──
# summary_df.to_csv(f"{OUTPUT_DIR}/basic_accuracy_summary.csv", index=False)
# print(f"✅ Summary saved to {OUTPUT_DIR}/basic_accuracy_summary.csv")









