import os
import re
import cohere
import pandas as pd
from collections import Counter
from time import sleep
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mmlu

co = cohere.ClientV2(API_KEY)
OUTPUT_DIR = mmlu.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_WORKERS = 10  # tune based on your Cohere rate limit tier

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


def parse_answer(cot_text: str, answer_prefix: str) -> tuple[str, str]:
    """
    Parse A/B/C/D answer from model output.
    Returns: (extracted_answer, parse_method)
    """
    lines = cot_text.strip().split("\n")

    # Primary: "the answer is (X)"
    match = re.search(rf"{answer_prefix}\s*\(([A-Da-d])\)", cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "primary_parentheses"

    # Fallback 1: "the answer is X" with colon
    match = re.search(rf"{answer_prefix}\s*[:：]\s*([A-Da-d])", cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "primary_colon"

    # Fallback 2: "the answer is X" with space
    match = re.search(rf"{answer_prefix}\s+([A-Da-d])(?:\s|$|\.)", cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "primary_space"

    # Fallback 3: standalone letter on last 3 lines — (A), [A], A., **A**, etc.
    for line in reversed(lines[-3:]):
        m = re.match(r"^[(\[\*]*([A-Da-d])[)\]\*\.]*\s*$", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper(), "last_line_letter"

    # Fallback 4: "Answer: X", "Final answer: X", "therefore the answer is X"
    for pattern in [
        r'(?:final\s+)?answer\s*[:：]\s*([A-Da-d])',
        r'(?:therefore|thus|so),?\s+(?:the\s+)?answer\s+is\s+([A-Da-d])',
        r'(?:option|choice)\s+([A-Da-d])',
    ]:
        match = re.search(pattern, cot_text, re.IGNORECASE)
        if match:
            return match.group(1).upper(), "answer_pattern"

    # Fallback 5: bold **A** or **Option A**
    match = re.search(r'\*\*\s*(?:option\s+)?([A-Da-d])\s*\*\*', cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "bold_answer"

    # Fallback 6: last standalone A/B/C/D in text
    matches = re.findall(r'\b([A-D])\b', cot_text.upper())
    if matches:
        return matches[-1], "last_letter"

    # Fallback 7: scan last line for any A-D
    if lines:
        last_line = lines[-1].upper()
        for letter in ['D', 'C', 'B', 'A']:
            if letter in last_line:
                return letter, "last_line_scan"

    return "", "failed"


def answers_are_equal(extracted: str, expected: str) -> bool:
    return str(extracted).strip().upper() == str(expected).strip().upper()


def call_with_retry(prompt: str, model=MODEL_NAME, max_retries=5) -> str:
    """Call Cohere API with retry on rate limits."""
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.6,
                # repetition_penalty=1.2,
            )
            return response.message.content[0].text.strip()
        except Exception as e:
            error_str = str(e).lower()
            if "rate" in error_str or "limit" in error_str or "429" in error_str:
                wait = 60 * (attempt + 1)
                safe_print(f"  ⏳ Rate limit. Waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                sleep(wait)
            else:
                raise e
    raise Exception(f"Failed after {max_retries} retries due to rate limiting")


def process_row(idx: int, row: pd.Series, instruction_template: str,
                answer_prefix: str, lang: str, debug: bool) -> dict:
    """Process a single question — called in parallel."""
    prompt = instruction_template.format(
        question=row["question"],
        A=row["A"],
        B=row["B"],
        C=row["C"],
        D=row["D"],
    )
    try:
        cot = call_with_retry(prompt)
        extracted, method = parse_answer(cot, answer_prefix)
        correct = answers_are_equal(extracted, str(row["answer"]))

        if debug and not correct:
            safe_print(f"\n  [DEBUG] [{lang.upper()}] Q{idx+1} INCORRECT:")
            safe_print(f"  Expected: {row['answer']}, Got: {extracted}, Method: {method}")
            safe_print(f"  CoT tail: ...{cot[-300:]}")
            safe_print(f"  {'-'*50}")

        safe_print(
            f"  [{lang.upper()}] Q{idx+1}: extracted={extracted!r:>3} | "
            f"expected={row['answer']} | method={method:20s} | {'✓' if correct else '✗'}"
        )

    except Exception as e:
        safe_print(f"  [{lang.upper()}] Q{idx+1} ERROR: {type(e).__name__}: {e}")
        cot, extracted, method, correct = "ERROR", "", "error", False

    return {
        "idx": idx,
        "cot": cot,
        "extracted": extracted,
        "parse_method": method,
        "correct": correct,
    }


def run_inference_for_lang(lang: str, debug: bool = False):
    """Run parallel single-pass inference for one language."""
    print(f"\n{'='*60}")
    print(f"Running MMLU inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    print(f"{'='*60}")

    df = pd.read_csv(mmlu.LANG_TO_DATA_PATH[lang])
    # df = df.head(20)  # uncomment for testing

    if "subject" not in df.columns:
        print("⚠️  Warning: 'subject' column not found")

    cols = ["question", "A", "B", "C", "D", "answer","subject"]
    # if "subject" in df.columns:
    #     cols.append("subject")
    df = df[cols].copy().reset_index(drop=True)

    print(f"Total questions: {len(df)}\n")

    instruction_template = mmlu.LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = mmlu.LANG_TO_ANSWER_PREFIX[lang]

    # --- Parallel inference across all questions ---
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(
                process_row, idx, row, instruction_template, answer_prefix, lang, debug
            ): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(futures):
            result = future.result()
            results[result["idx"]] = result

    # --- Reconstruct in original question order ---
    cots              = [results[i]["cot"]          for i in range(len(df))]
    extracted_answers = [results[i]["extracted"]    for i in range(len(df))]
    parse_methods     = [results[i]["parse_method"] for i in range(len(df))]
    corrects          = [results[i]["correct"]      for i in range(len(df))]

    df["cot"]          = cots
    df["extracted"]    = extracted_answers
    df["parse_method"] = parse_methods
    df["correct"]      = corrects

    # --- Parse method breakdown ---
    method_counts = Counter(parse_methods)
    total_calls = len(df)
    failed_calls = method_counts.get("failed", 0)

    print(f"\n{'─'*50}")
    print(f"[{lang.upper()}] PARSE METHOD BREAKDOWN ({total_calls} total calls):")
    for method, count in method_counts.most_common():
        print(f"  {method:25s}: {count:4d}  ({count/total_calls*100:.1f}%)")
    print(f"  → Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    # --- Overall accuracy ---
    accuracy = sum(corrects) / total_calls
    print(f"\n[{lang.upper()}] ACCURACY: {sum(corrects)}/{total_calls} = {accuracy:.1%}")

    # --- Subject-level accuracy ---
    if "subject" in df.columns:
        print(f"\n[{lang.upper()}] ACCURACY BY SUBJECT (top 10 by count):")
        subj = df.groupby("subject")["correct"].agg(["sum", "count"])
        subj["accuracy"] = subj["sum"] / subj["count"]
        subj = subj.sort_values("count", ascending=False).head(10)
        for subj_name, row in subj.iterrows():
            print(f"  {subj_name:45s}: {row['sum']:4.0f}/{row['count']:4.0f} = {row['accuracy']:.1%}")

    # --- Collect failed parse rows ---
    failed_rows = []
    for i, row in df.iterrows():
        if parse_methods[i] == "failed":
            failed_rows.append({
                "lang": lang,
                "q_idx": i + 1,
                "question": row["question"][:100],
                "expected": row["answer"],
                "cot_tail": "\n".join(cots[i].strip().split("\n")[-6:]),
            })

    if failed_rows:
        print(f"\n[{lang.upper()}] ❌ FAILED PARSE DETAILS:")
        for fr in failed_rows:
            print(f"  Q{fr['q_idx']} | expected={fr['expected']}")
            print(f"  Question: {fr['question']}")
            print(f"  --- Last 6 lines of CoT ---")
            print(fr["cot_tail"])
            print(f"  {'─'*40}")

    failed_df = pd.DataFrame(failed_rows) if failed_rows else pd.DataFrame()

    # --- Save per-language CSV ---
    csv_path = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
    df.to_csv(csv_path, index=False)
    print(f"✅ Saved: {csv_path}")

    # --- Save parse failures to xlsx if any ---
    if not failed_df.empty:
        fail_path = os.path.join(OUTPUT_DIR, f"parse_failures_{lang}.xlsx")
        failed_df.to_excel(fail_path, index=False)
        print(f"⚠️  Parse failures saved: {fail_path}")

    return lang, sum(corrects), total_calls, failed_rows, df, failed_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Show CoT for incorrect answers")
    args = parser.parse_args()

    summary = []
    all_failed_parses = []
    lang_dfs = {}

    for lang in ALL_LANGUAGES:
        lang, correct, total, failed_rows, df, failed_df = run_inference_for_lang(lang, debug=args.debug)
        summary.append({
            "language": LANG_TO_FULL_NAME[lang],
            "code": lang,
            "correct": correct,
            "total": total,
            "accuracy": f"{correct/total:.1%}",
            "parse_failures": len(failed_rows),
        })
        all_failed_parses.extend(failed_rows)
        lang_dfs[lang] = {"df": df, "failed_df": failed_df}

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    summary_df = pd.DataFrame(summary).sort_values("language")
    print(summary_df.to_string(index=False))
    print(f"\nTotal parse failures across all languages: {sum(r['parse_failures'] for r in summary)}")

    # --- Save summary xlsx ---
    summary_path = os.path.join(OUTPUT_DIR, "summary.xlsx")
    all_failed_df = pd.DataFrame(all_failed_parses) if all_failed_parses else pd.DataFrame()
    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not all_failed_df.empty:
            all_failed_df.to_excel(writer, sheet_name="all_parse_failures", index=False)
    print(f"\n✅ Summary saved: {summary_path}")