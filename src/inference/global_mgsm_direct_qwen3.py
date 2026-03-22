import os
import re
import csv
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.configs.global_mgsm_qwen3 import (
    QWEN3_MODEL_NAME, ALL_GLOBAL_MGSM_LANGUAGES, QWEN3_MGSM_DIRECT_DIR,
)
from src.configs import mgsm

OUTPUT_DIR = QWEN3_MGSM_DIRECT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

CKPT_COLS = ["question", "answer", "raw_answer", "extracted_answer", "is_correct"]

# Load model once
print(f"Loading model: {QWEN3_MODEL_NAME}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL_NAME, dtype=torch.float16)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")


def parse_answer(response_text: str, answer_prefix: str) -> str:
    if not response_text or response_text.strip() == "":
        return ""
    text = response_text.replace(",", "").strip()
    if re.fullmatch(r"\d+\.?\d*", text):
        return text.rstrip(".")
    for prefix in [answer_prefix, answer_prefix.rstrip(":") + ":"]:
        if prefix in response_text:
            after = response_text.split(prefix)[-1].strip()
            numbers = re.findall(r"\d+\.?\d*", after.replace(",", ""))
            if numbers:
                return numbers[0].rstrip(".")
    numbers = re.findall(r"\d+\.?\d*", text)
    return numbers[-1].rstrip(".") if numbers else ""


def answers_are_equal(extracted: str, expected: str) -> bool:
    try:
        return float(str(extracted).replace(",", "").strip()) == \
               float(str(expected).replace(",", "").strip())
    except (ValueError, TypeError):
        return False


def call_model(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def run_inference_for_lang(lang: str):
    final_path = os.path.join(OUTPUT_DIR, f"direct_{lang}.xlsx")
    ckpt_path  = os.path.join(OUTPUT_DIR, f"direct_{lang}_checkpoint.csv")

    # Skip if final xlsx already exists and is complete
    if os.path.exists(final_path):
        existing = pd.read_excel(final_path, sheet_name="results")
        if len(existing) >= 250:
            print(f"[{lang.upper()}] Already complete ({len(existing)} rows) — skipping.")
            return lang, existing["is_correct"].sum(), len(existing)

    # Load checkpoint if it exists
    completed_indices = set()
    if os.path.exists(ckpt_path):
        ckpt_df = pd.read_csv(ckpt_path)
        completed_indices = set(range(len(ckpt_df)))
        print(f"[{lang.upper()}] Resuming from Q{len(ckpt_df)+1} ({len(ckpt_df)} rows already done)")

    print(f"\n{'='*60}")
    print(f"Running MGSM DIRECT inference [{QWEN3_MODEL_NAME}]: {lang.upper()}")
    print(f"{'='*60}")

    df = pd.read_csv(mgsm.LANG_TO_DATA_PATH[lang])
    df = df[["question", "answer"]].copy()

    instruction_template = mgsm.LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = mgsm.LANG_TO_ANSWER_PREFIX[lang]

    # Open checkpoint for appending
    write_header = not os.path.exists(ckpt_path)
    ckpt_file = open(ckpt_path, "a", newline="", encoding="utf-8")
    ckpt_writer = csv.DictWriter(ckpt_file, fieldnames=CKPT_COLS, quoting=csv.QUOTE_ALL)
    if write_header:
        ckpt_writer.writeheader()

    for idx, row in df.iterrows():
        if idx in completed_indices:
            continue

        question = row["question"]
        print(f"\nQ{idx+1}: {str(question)[:80]}...")
        prompt = instruction_template.format(input=question)

        try:
            raw = call_model(prompt)
            print(f"  Raw     : {raw!r}")
            extracted = parse_answer(raw, answer_prefix)
            correct = answers_are_equal(extracted, str(row["answer"]))
            match = "OK" if correct else "WRONG"
            print(f"  [{match}] Extracted: {extracted!r} | Expected: {row['answer']}")
        except Exception as e:
            print(f"  ERROR on Q{idx+1}: {type(e).__name__}: {e}")
            raw = "ERROR"
            extracted = ""
            correct = False

        ckpt_writer.writerow({
            "question": question,
            "answer": row["answer"],
            "raw_answer": raw,
            "extracted_answer": extracted,
            "is_correct": correct,
        })
        ckpt_file.flush()

    ckpt_file.close()

    # Build final dataframe from checkpoint
    result_df = pd.read_csv(ckpt_path)

    accuracy = result_df["is_correct"].sum()
    total = len(result_df)
    print(f"\n[{lang.upper()}] Accuracy: {accuracy}/{total} = {accuracy/total:.1%}")

    failures = result_df[~result_df["is_correct"]][["question", "answer", "raw_answer", "extracted_answer"]]

    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="results", index=False)
        if not failures.empty:
            failures.to_excel(writer, sheet_name="failures", index=False)
    print(f"Saved: {final_path}")

    # Remove checkpoint now that xlsx is written
    os.remove(ckpt_path)

    return lang, accuracy, total


if __name__ == "__main__":
    summary = []
    for lang in ALL_GLOBAL_MGSM_LANGUAGES:
        lang, correct, total = run_inference_for_lang(lang)
        summary.append({
            "language": lang,
            "correct": correct,
            "total": total,
            "accuracy": f"{correct/total:.1%}",
        })

    print(f"\n{'='*60}")
    print(f"FINAL ACCURACY SUMMARY — MGSM Direct ({QWEN3_MODEL_NAME})")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, "summary_direct.xlsx")
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
    print(f"Summary saved: {summary_path}")
