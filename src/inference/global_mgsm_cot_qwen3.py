import os
import re
import csv
import threading
import pandas as pd
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from src.configs.global_mgsm_qwen3 import (
    QWEN3_MODEL_NAME, ALL_GLOBAL_MGSM_LANGUAGES, QWEN3_MGSM_COT_DIR,
)
from src.configs import mgsm

OUTPUT_DIR = QWEN3_MGSM_COT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 3
MAX_WORKERS = 3
LANG_WORKERS = 1

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# Load model once
print(f"Loading model: {QWEN3_MODEL_NAME}")
device = "mps" if torch.backends.mps.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(QWEN3_MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(QWEN3_MODEL_NAME, dtype=torch.float16)
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

model_lock = threading.Lock()


def call_model(prompt: str) -> str:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(device)
    with model_lock:
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    generated = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_answer(cot_text: str, answer_prefix: str, lang: str = "") -> tuple[str, str]:
    lines = cot_text.strip().split("\n")

    def extract_numbers(text):
        return re.findall(r"\d+\.?\d*", text.replace(",", ""))

    if lang == "zh":
        def normalize_colons(s):
            return s.replace("：", ":").replace("︓", ":")

        norm_prefix = normalize_colons(answer_prefix)

        for line in reversed(lines):
            clean = line.replace("**", "").strip()
            if answer_prefix in clean:
                nums = extract_numbers(clean)
                if nums:
                    return nums[-1].rstrip("."), "primary"

        for line in reversed(lines):
            clean = line.replace("**", "").strip()
            if norm_prefix in normalize_colons(clean):
                nums = extract_numbers(clean)
                if nums:
                    return nums[-1].rstrip("."), "zh_fuzzy"

        for line in reversed(lines):
            if re.search(r'\*\*.*答案.*\*\*', line):
                nums = extract_numbers(line)
                if nums:
                    return nums[-1].rstrip("."), "zh_bold_prefix"

        def extract_boxed_zh(line):
            m = re.search(r'\\boxed\{([^}]+)\}', line)
            if not m:
                return ""
            inner = m.group(1)
            if "=" in inner:
                inner = inner.split("=")[-1]
            nums = re.findall(r"\d+\.?\d*", inner.replace(",", ""))
            return nums[-1].rstrip(".") if nums else ""

        for i, line in enumerate(lines):
            clean = line.replace("**", "").strip()
            if norm_prefix in normalize_colons(clean) and i + 1 < len(lines):
                val = extract_boxed_zh(lines[i + 1])
                if val:
                    return val, "next_line_box"

        for i, line in enumerate(lines):
            clean = line.replace("**", "").strip()
            if norm_prefix in normalize_colons(clean) and i + 1 < len(lines):
                next_line = lines[i + 1].replace("**", "").strip()
                next_clean = next_line.replace("$", "").replace("₹", "").replace("%", "")
                nums = extract_numbers(next_clean)
                if nums:
                    return nums[0].rstrip("."), "next_line_num"

        for line in reversed(lines[-5:]):
            val = extract_boxed_zh(line)
            if val:
                return val, "fallback_box"

        all_nums = extract_numbers(cot_text)
        if all_nums:
            return all_nums[-1].rstrip("."), "last_number"

        return "", "failed"

    def extract_boxed(line):
        m = re.search(r'\\boxed\{([^}]+)\}', line)
        if not m:
            return ""
        inner = m.group(1)
        if "=" in inner:
            inner = inner.split("=")[-1]
        nums = re.findall(r"\d+\.?\d*", inner.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""

    def extract_inline_latex(line):
        m = re.search(r'\\\(([^)]+)\\\)|\\\[([^\]]+)\\\]', line)
        if not m:
            return ""
        inner = (m.group(1) or m.group(2) or "").strip()
        nums = re.findall(r"\d+\.?\d*", inner.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""

    for line in reversed(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line:
            numbers = re.findall(r"\d+\.?\d*", clean_line.replace(",", ""))
            if numbers:
                return numbers[-1].rstrip("."), "primary"

    for i, line in enumerate(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line and i + 1 < len(lines):
            val = extract_boxed(lines[i + 1])
            if val:
                return val, "next_line_box"

    for i, line in enumerate(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line and i + 1 < len(lines):
            next_line = lines[i + 1].replace("**", "").strip()
            next_clean = next_line.replace("$", "").replace("₹", "").replace("%", "")
            numbers = re.findall(r"\d+\.?\d*", next_clean.replace(",", ""))
            if numbers:
                return numbers[0].rstrip("."), "next_line_num"

    for line in reversed(lines[-5:]):
        val = extract_boxed(line)
        if val:
            return val, "fallback_box"

    for line in reversed(lines[-5:]):
        val = extract_inline_latex(line)
        if val:
            return val, "fallback_inline_latex"

    for line in reversed(lines[-5:]):
        bold_nums = re.findall(r'\*\*[^*]*?(\d+\.?\d*)[^*]*?\*\*', line.replace(",", ""))
        if bold_nums:
            return bold_nums[-1].rstrip("."), "fallback_bold"

    all_nums = re.findall(r"\d+\.?\d*", cot_text.replace(",", ""))
    if all_nums:
        return all_nums[-1].rstrip("."), "last_number"

    return "", "failed"


def answers_are_equal(extracted: str, expected: str) -> bool:
    try:
        return float(str(extracted).replace(",", "").strip()) == \
               float(str(expected).replace(",", "").strip())
    except (ValueError, TypeError):
        return False


def majority_vote(answers: list[str]) -> tuple[str, str]:
    valid = [a for a in answers if str(a).strip()]
    if not valid:
        return "", "no_answer"
    counts = Counter(valid)
    top_answer, top_count = counts.most_common(1)[0]
    if top_count == len(answers):
        return top_answer, "unanimous"
    elif top_count > 1:
        return top_answer, "majority"
    else:
        return valid[0], "all_differ"


def fetch_single_run(run_idx: int, prompt: str, answer_prefix: str, expected: str, q_label: str, lang: str = ""):
    try:
        cot = call_model(prompt)
        extracted, method = parse_answer(cot, answer_prefix, lang)
        correct = answers_are_equal(extracted, expected)
        safe_print(
            f"  {q_label} Run {run_idx+1}: "
            f"extracted={extracted!r:>10} | expected={expected} | "
            f"method={method} | {'OK' if correct else 'WRONG'}"
        )
        return {"run": run_idx, "cot": cot, "extracted": extracted, "parse_method": method, "correct": correct}
    except Exception as e:
        safe_print(f"  {q_label} Run {run_idx+1} ERROR: {type(e).__name__}: {e}")
        return {"run": run_idx, "cot": "ERROR", "extracted": "", "parse_method": "error", "correct": False}


# Checkpoint columns
CKPT_COLS = (
    ["question", "answer"] +
    [f"cot_run{r+1}" for r in range(N_RUNS)] +
    [f"extracted_run{r+1}" for r in range(N_RUNS)] +
    [f"parse_method_run{r+1}" for r in range(N_RUNS)] +
    [f"correct_run{r+1}" for r in range(N_RUNS)] +
    ["majority_vote", "vote_status", "majority_correct"]
)


def run_inference_for_lang(lang: str):
    final_path = os.path.join(OUTPUT_DIR, f"cot_majority_{lang}.xlsx")
    ckpt_path  = os.path.join(OUTPUT_DIR, f"cot_majority_{lang}_checkpoint.csv")

    # Skip if final xlsx already exists and is complete
    if os.path.exists(final_path):
        existing = pd.read_excel(final_path, sheet_name="results")
        if len(existing) >= 250:
            safe_print(f"[{lang.upper()}] Already complete ({len(existing)} rows) — skipping.")
            return lang, existing["majority_correct"].sum(), len(existing), [], existing, pd.DataFrame()

    # Load checkpoint if it exists
    completed_indices = set()
    if os.path.exists(ckpt_path):
        ckpt_df = pd.read_csv(ckpt_path)
        completed_indices = set(range(len(ckpt_df)))
        safe_print(f"[{lang.upper()}] Resuming from Q{len(ckpt_df)+1} ({len(ckpt_df)} rows already done)")

    safe_print(f"\n{'='*60}")
    safe_print(f"Running MGSM {N_RUNS}-run majority vote [{QWEN3_MODEL_NAME}]: {lang.upper()}")
    safe_print(f"{'='*60}")

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
        q_label = f"[{lang.upper()}] Q{idx+1}"
        safe_print(f"\n{q_label}: {str(question)[:80]}...")
        prompt = instruction_template.format(input=question)

        run_results = [None] * N_RUNS
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_run = {
                executor.submit(fetch_single_run, run, prompt, answer_prefix, str(row["answer"]), q_label, lang): run
                for run in range(N_RUNS)
            }
            for future in as_completed(future_to_run):
                result = future.result()
                run_results[result["run"]] = result

        answers_this_q = [run_results[r]["extracted"] for r in range(N_RUNS)]
        vote, status = majority_vote(answers_this_q)
        flag = " (ALL DIFFER — falling back to run 1)" if status == "all_differ" else ""
        safe_print(f"  -> [{status}] vote={vote!r} | expected={row['answer']}{flag}")

        ckpt_row = {
            "question": question,
            "answer": row["answer"],
            **{f"cot_run{r+1}": run_results[r]["cot"] for r in range(N_RUNS)},
            **{f"extracted_run{r+1}": run_results[r]["extracted"] for r in range(N_RUNS)},
            **{f"parse_method_run{r+1}": run_results[r]["parse_method"] for r in range(N_RUNS)},
            **{f"correct_run{r+1}": run_results[r]["correct"] for r in range(N_RUNS)},
            "majority_vote": vote,
            "vote_status": status,
            "majority_correct": answers_are_equal(vote, str(row["answer"])),
        }
        ckpt_writer.writerow(ckpt_row)
        ckpt_file.flush()

    ckpt_file.close()

    # Build final dataframe from checkpoint
    result_df = pd.read_csv(ckpt_path)

    all_methods = []
    for r in range(N_RUNS):
        all_methods.extend(result_df[f"parse_method_run{r+1}"].tolist())
    method_counts = Counter(all_methods)
    total_calls = len(result_df) * N_RUNS
    failed_calls = method_counts.get("failed", 0)

    safe_print(f"\n{'─'*50}")
    safe_print(f"[{lang.upper()}] PARSE METHOD BREAKDOWN ({total_calls} total calls):")
    for method, count in method_counts.most_common():
        safe_print(f"  {method:20s}: {count:3d}  ({count/total_calls*100:.1f}%)")
    safe_print(f"  -> Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    failed_rows = []
    for i, row in result_df.iterrows():
        for r in range(N_RUNS):
            if row[f"parse_method_run{r+1}"] == "failed":
                failed_rows.append({
                    "lang": lang,
                    "q_idx": i + 1,
                    "run": r + 1,
                    "question": str(row["question"])[:100],
                    "expected": row["answer"],
                    "cot_tail": "\n".join(str(row[f"cot_run{r+1}"]).strip().split("\n")[-6:]),
                })

    total = len(result_df)
    safe_print(f"\n[{lang.upper()}] ACCURACY:")
    for r in range(N_RUNS):
        acc = result_df[f"correct_run{r+1}"].sum()
        safe_print(f"  Run {r+1}: {acc}/{total} = {acc/total:.1%}")
    majority_acc = result_df["majority_correct"].sum()
    safe_print(f"  Majority vote: {majority_acc}/{total} = {majority_acc/total:.1%}")

    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        result_df.to_excel(writer, sheet_name="results", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="parse_failures", index=False)
    safe_print(f"Saved: {final_path}")

    # Remove checkpoint now that xlsx is written
    os.remove(ckpt_path)

    return lang, majority_acc, total, failed_rows, result_df, failed_df


if __name__ == "__main__":
    summary = []
    all_failed_parses = []
    lang_results = {}

    with ThreadPoolExecutor(max_workers=LANG_WORKERS) as executor:
        future_to_lang = {
            executor.submit(run_inference_for_lang, lang): lang
            for lang in ALL_GLOBAL_MGSM_LANGUAGES
        }
        for future in as_completed(future_to_lang):
            lang, correct, total, failed_rows, df, failed_df = future.result()
            summary.append({
                "language": lang,
                "correct": correct,
                "total": total,
                "accuracy": f"{correct/total:.1%}",
                "parse_failures": len(failed_rows),
            })
            all_failed_parses.extend(failed_rows)
            lang_results[lang] = {"df": df, "failed_df": failed_df}

    print(f"\n{'='*60}")
    print(f"SUMMARY (majority vote) — {QWEN3_MODEL_NAME}")
    print(f"{'='*60}")
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, "summary_majority.xlsx")
    all_failed_df = pd.DataFrame(all_failed_parses) if all_failed_parses else pd.DataFrame()
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not all_failed_df.empty:
            all_failed_df.to_excel(writer, sheet_name="all_parse_failures", index=False)
    print(f"Summary saved: {summary_path}")

    final_path = os.path.join(OUTPUT_DIR, "final_results.xlsx")
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for lang in sorted(lang_results.keys()):
            lang_results[lang]["df"].to_excel(writer, sheet_name=f"results_{lang}"[:31], index=False)
            if not lang_results[lang]["failed_df"].empty:
                lang_results[lang]["failed_df"].to_excel(writer, sheet_name=f"failures_{lang}"[:31], index=False)
    print(f"Combined final file saved: {final_path}")
