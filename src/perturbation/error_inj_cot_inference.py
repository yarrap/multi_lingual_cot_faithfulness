import os
import re
import pandas as pd
import cohere
from time import sleep
from openpyxl import load_workbook
import sys

# 1. Calculate the path to the project root (2 levels up from src/inference/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Add that root to Python's search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs import (
    API_KEY,
    MODEL_NAME,
    mgsm
)

co = cohere.ClientV2(API_KEY)


# -------------------------------------------------------
# Answer Parser
# -------------------------------------------------------

def parse_answer(response_text: str, lang: str) -> str:

    if not response_text:
        return ""

    text = str(response_text).replace(",", "").strip()

    # Stage 1: pure number
    if re.fullmatch(r"-?\d+\.?\d*", text):
        return text.rstrip(".")

    # Stage 2: native prefix
    prefixes = mgsm.LANG_TO_ANSWER_PREFIX[lang]
    for prefix in prefixes:
        if prefix in response_text:
            after = response_text.split(prefix)[-1].strip()
            numbers = re.findall(r"-?\d+\.?\d*", after)
            if numbers:
                return numbers[0].rstrip(".")

    # Stage 3: fallback
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1].rstrip(".") if numbers else ""


# -------------------------------------------------------
# Model Caller
# -------------------------------------------------------

def call_with_retry(prompt: str, max_retries: int = 5) -> str:

    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0,
            )
            return response.message.content[0].text.strip()

        except Exception as e:
            if "rate" in str(e).lower():
                wait = 60 * (attempt + 1)
                print(f"Rate limit. Waiting {wait}s...")
                sleep(wait)
            else:
                raise e

    return ""


# -------------------------------------------------------
# Append Raw + Parsed Reinference
# -------------------------------------------------------


def append_reinference_to_excel():

    file_path = "results/hypothesis_2/error_inj_perturbation/mgsm/final_data_error_inj_tiny-aya-global.xlsx"
    languages = ["en", "bn", "sw", "te", "zh"]

    for lang in languages:

        sheet_name = f"cot_majority_{lang}"
        print(f"\nProcessing {sheet_name}")

        df = pd.read_excel(file_path, sheet_name=sheet_name).fillna("")

        template = mgsm.LANG_TO_PERTURBATION_PROMPT[lang]
        instruction_template = mgsm.LANG_TO_QUESTION_TEMPLATE[lang]

        raw_outputs = []
        parsed_answers = []

        for idx, row in df.iterrows():

            question_prompt = instruction_template.format(
                question=row["question"]
            )

            prompt = template.format(
                question_prompt=question_prompt,
                cot_variant=row["error_inj_cot"]
            )

            raw_resp = call_with_retry(prompt)
            extracted = parse_answer(raw_resp, lang)

            raw_outputs.append(raw_resp)
            parsed_answers.append(extracted)

            sleep(1.0)

        df["re_infer_raw"] = raw_outputs
        df["re_infer_answer"] = parsed_answers

        # Save/replace sheet in-place
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)

    print("\n Reinference raw + answer appended successfully.")

if __name__ == "__main__":
    append_reinference_to_excel()