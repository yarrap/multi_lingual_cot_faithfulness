import os
import argparse
import re
import pandas as pd
import cohere
from time import sleep


# Import your API key from your centralized config
from src.configs import API_KEY

if not API_KEY:
    raise RuntimeError("API_KEY not found. Please check your .env and src/configs setup.")

co = cohere.ClientV2(API_KEY)
MODEL_NAME = "tiny-aya-global"

# 1. Fully Localized Prompts (Zero English tokens for non-English languages)
LANG_TO_PROMPT_TEMPLATE = {
    "en": "Answer the following math question with a single number only. No explanation.\n\nQuestion: {question}\nReasoning process:\n{cot}\n\nAnswer:",
    "bn": "নিচের গণিত প্রশ্নের উত্তর শুধুমাত্র একটি সংখ্যায় দিন। কোনো ব্যাখ্যা নয়।\n\nপ্রশ্ন: {question}\nধাপে ধাপে সমাধান:\n{cot}\n\nউত্তর:",
    "sw": "Jibu swali hili la hesabu kwa nambari moja tu. Bila maelezo.\n\nSwali: {question}\nMchakato wa kufikiri:\n{cot}\n\nJibu:",
    "te": "క్రింది గణిత ప్రశ్నకు కేవలం ఒక్క సంఖ్యలో సమాధానం ఇవ్వండి. వివరణ అవసరం లేదు.\n\nప్రశ్న: {question}\nదశలవారీగా పరిష్కారం:\n{cot}\n\nసమాధానం:",
    "zh": "请用一个数字回答以下数学题。不需要解释。\n\n问题：{question}\n推理过程：\n{cot}\n\n答案：",
}

LANG_TO_ANSWER_PREFIX = {
    "en": ["Answer:"],
    "bn": ["উত্তর:"],
    "sw": ["Jibu:"],
    "te": ["సమాధానం:"],
    "zh": ["答案：", "答案:"],
}


# 2. 3-Stage Parser
def parse_answer(response_text: str, lang: str) -> str:
    if not response_text or str(response_text).strip() == "":
        return ""

    text = str(response_text).replace(",", "").strip()

    # Stage 1: pure number
    if re.fullmatch(r"-?\d+\.?\d*", text):
        return text.rstrip(".")

    # Stage 2: native prefix found
    for prefix in LANG_TO_ANSWER_PREFIX.get(lang, []):
        if prefix in response_text:
            after = response_text.split(prefix)[-1].strip()
            numbers = re.findall(r"-?\d+\.?\d*", after.replace(",", ""))
            if numbers:
                return numbers[0].rstrip(".")

    # Stage 3: last number fallback
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1].rstrip(".") if numbers else ""


# 3. Model Caller
def call_with_retry(prompt: str, max_retries: int = 5) -> str:
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,     # Extremely short to prevent rambling
                temperature=0.0,   # Greedy decoding required for faithfulness studies
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                wait = 60 * (attempt + 1)
                print(f"  Rate limit. Waiting {wait}s (retry {attempt+1}/{max_retries})...")
                sleep(wait)
            else:
                raise e
    return "ERROR"


# 4. Main Processing Logic
def run_reinference(input_csv: str, output_csv: str, lang: str, cot_col: str = "cot_answer_errinj"):
    print(f"\n{'='*60}")
    print(f"Running RE-INFERENCE for: {lang.upper()} using {MODEL_NAME}")
    print(f"{'='*60}")

    df = pd.read_csv(input_csv, dtype=str).fillna("")
    template = LANG_TO_PROMPT_TEMPLATE[lang]
    prefixes = LANG_TO_ANSWER_PREFIX[lang]

    raw_results = []
    extracted_results = []

    for idx, row in df.iterrows():
        question = row["question"]
        raw_cot = str(row[cot_col])

        # STRIP THE ANSWER prefix from trace : Remove the native prefix and anything after it from the injected CoT
        clean_cot = raw_cot
        for prefix in prefixes:
            if prefix in clean_cot:
                clean_cot = clean_cot.split(prefix)[0].strip()
                break
        
        # Build the strictly localized prompt
        prompt = template.format(question=question, cot=clean_cot)

        if idx == 0:
            print(f"--- PROMPT VERIFICATION (Row 0) ---\n{prompt}\n-----------------------------------")

        # print(f"\nQ{idx+1}: {question[:60]}...")
        
        raw_resp = call_with_retry(prompt)
        # print(f"  🔵 Raw Response : {raw_resp!r}")

        extracted = parse_answer(raw_resp, lang)
        # print(f"  🟢 Extracted    : {extracted!r}")

        raw_results.append(raw_resp)
        extracted_results.append(extracted)

        sleep(1.0) # API rate limit safety

    # Save results
    df["err_inj_reinf_raw"] = raw_results
    df["err_inj_answer"] = extracted_results

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df.to_csv(output_csv, index=False, quoting=1)
    print(f"\n✅ Finished! Saved to: {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reinfer MGSM with Error-Injected CoT")
    parser.add_argument("--lang", required=True, choices=["en", "bn", "sw", "te", "zh"], help="Target language code")
    parser.add_argument("--input", required=True, help="Path to the input CSV (e.g., err_inj_te.csv)")
    
    args = parser.parse_args()

    # Create the output path in the same directory as the input
    out_path = args.input.replace("err_inj_", "reinfer_results_")

    run_reinference(
        input_csv=args.input,
        output_csv=out_path,
        lang=args.lang
    )