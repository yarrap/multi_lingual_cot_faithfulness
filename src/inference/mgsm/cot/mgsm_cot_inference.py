import os
import re
import cohere
import pandas as pd
from collections import Counter
from time import sleep
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..","..","..")))
from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mgsm

co = cohere.ClientV2(API_KEY)
OUTPUT_DIR = mgsm.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 3
MAX_WORKERS = 3
LANG_WORKERS = 1

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


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

    # DEFAULT — all languages except zh

    def extract_boxed(line):
        """
        Extract a number from \\boxed{...} handling:
          \\boxed{34}          → '34'
          \\boxed{105.83}      → '105.83'   FIX: decimals
          \\boxed{6, 8, 20}    → '20'       FIX: multiple nums → take last
          \\boxed{6+8+20=34}   → '34'       FIX: expression → take number after =
          \\boxed{10W}         → ''         non-numeric → skip
        """
        m = re.search(r'\\boxed\{([^}]+)\}', line)
        if not m:
            return ""
        inner = m.group(1)
        # expression with = : take the number after the last =
        if "=" in inner:
            inner = inner.split("=")[-1]
        # extract all numbers (including decimals)
        nums = re.findall(r"\d+\.?\d*", inner.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""

    def extract_inline_latex(line):
        """
        Extract number from inline LaTeX: \\( 24 \\) or \\[ 24 \\]
        FIX: catches \\( 24 \\) pattern missed by all previous rules
        """
        m = re.search(r'\\\(([^)]+)\\\)|\\\[([^\]]+)\\\]', line)
        if not m:
            return ""
        inner = (m.group(1) or m.group(2) or "").strip()
        nums = re.findall(r"\d+\.?\d*", inner.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""

    # Primary: answer_prefix + number on the same line
    for line in reversed(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line:
            numbers = re.findall(r"\d+\.?\d*", clean_line.replace(",", ""))
            if numbers:
                return numbers[-1].rstrip("."), "primary"

    # Fallback 1a: prefix on its own line → \boxed{} on next line
    for i, line in enumerate(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line and i + 1 < len(lines):
            val = extract_boxed(lines[i + 1])
            if val:
                return val, "next_line_box"

    # Fallback 1b: prefix on its own line → plain number on next line
    for i, line in enumerate(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line and i + 1 < len(lines):
            next_line = lines[i + 1].replace("**", "").strip()
            next_clean = next_line.replace("$", "").replace("₹", "").replace("%", "")
            numbers = re.findall(r"\d+\.?\d*", next_clean.replace(",", ""))
            if numbers:
                return numbers[0].rstrip("."), "next_line_num"

    # Fallback 2: \boxed{} anywhere in last 5 lines (with improved extractor)
    for line in reversed(lines[-5:]):
        val = extract_boxed(line)
        if val:
            return val, "fallback_box"

    # Fallback 3: inline LaTeX \( num \) anywhere in last 5 lines
    # FIX: catches \\( 24 \\) pattern (e.g. Bengali/Telugu)
    for line in reversed(lines[-5:]):
        val = extract_inline_latex(line)
        if val:
            return val, "fallback_inline_latex"

    # Fallback 4: bold answer pattern **number** or **number unit**
    # FIX: catches **13 বছর**, **-$10,000**, **70** etc. in last 5 lines
    # strip commas before matching so **-$10,000** → **-$10000** → 10000
    for line in reversed(lines[-5:]):
        bold_nums = re.findall(r'\*\*[^*]*?(\d+\.?\d*)[^*]*?\*\*', line.replace(",", ""))
        if bold_nums:
            return bold_nums[-1].rstrip("."), "fallback_bold"

    # Fallback 5: last number in entire CoT
    # FIX: catches truncated CoTs (repetition loop / max_tokens hit) where
    # the model never wrote the answer prefix line
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


def call_with_retry(prompt: str, model=MODEL_NAME, max_retries=5) -> str:
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=4096,
                temperature=0.6,
            )
            # 2048
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


def fetch_single_run(run_idx: int, prompt: str, answer_prefix: str, expected: str, q_label: str, lang: str = ""):
    try:
        cot = call_with_retry(prompt)
        extracted, method = parse_answer(cot, answer_prefix, lang)
        correct = answers_are_equal(extracted, expected)
        safe_print(
            f"  {q_label} Run {run_idx+1}: "
            f"extracted={extracted!r:>10} | expected={expected} | "
            f"method={method} | {'✓' if correct else '✗'}"
        )
        return {
            "run": run_idx,
            "cot": cot,
            "extracted": extracted,
            "parse_method": method,
            "correct": correct,
            "error": None,
        }
    except Exception as e:
        safe_print(f"  {q_label} Run {run_idx+1} ERROR: {type(e).__name__}: {e}")
        return {
            "run": run_idx,
            "cot": "ERROR",
            "extracted": "",
            "parse_method": "error",
            "correct": False,
            "error": str(e),
        }


def run_inference_for_lang(lang: str):
    safe_print(f"\n{'='*60}")
    safe_print(f"Running MGSM {N_RUNS}-run majority vote for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    safe_print(f"{'='*60}")

    df = pd.read_csv(mgsm.LANG_TO_DATA_PATH[lang])
    df = df[["question", "answer"]].copy()
    # df = df.head(10)

    instruction_template = mgsm.LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = mgsm.LANG_TO_ANSWER_PREFIX[lang]

    run_data = {run: {"cots": [], "extracted": [], "parse_methods": [], "corrects": []}
                for run in range(N_RUNS)}

    for idx, row in df.iterrows():
        question = row["question"]
        q_label = f"[{lang.upper()}] Q{idx+1}"
        safe_print(f"\n{q_label}: {question[:80]}...")
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

        for run in range(N_RUNS):
            r = run_results[run]
            run_data[run]["cots"].append(r["cot"])
            run_data[run]["extracted"].append(r["extracted"])
            run_data[run]["parse_methods"].append(r["parse_method"])
            run_data[run]["corrects"].append(r["correct"])

        answers_this_q = [run_results[run]["extracted"] for run in range(N_RUNS)]
        vote, status = majority_vote(answers_this_q)
        flag = " ⚠️  ALL DIFFER — falling back to run 1" if status == "all_differ" else ""
        safe_print(f"  → [{status}] vote={vote!r} | expected={row['answer']}{flag}")

    for run in range(N_RUNS):
        df[f"cot_run{run+1}"]          = run_data[run]["cots"]
        df[f"extracted_run{run+1}"]    = run_data[run]["extracted"]
        df[f"parse_method_run{run+1}"] = run_data[run]["parse_methods"]
        df[f"correct_run{run+1}"]      = run_data[run]["corrects"]

    vote_results = [
        majority_vote([run_data[run]["extracted"][i] for run in range(N_RUNS)])
        for i in range(len(df))
    ]
    df["majority_vote"]    = [v for v, _ in vote_results]
    df["vote_status"]      = [s for _, s in vote_results]
    df["majority_correct"] = df.apply(
        lambda r: answers_are_equal(r["majority_vote"], r["answer"]), axis=1
    )

    all_methods = []
    for run in range(N_RUNS):
        all_methods.extend(run_data[run]["parse_methods"])

    method_counts = Counter(all_methods)
    total_calls = len(df) * N_RUNS
    failed_calls = method_counts.get("failed", 0)

    safe_print(f"\n{'─'*50}")
    safe_print(f"[{lang.upper()}] PARSE METHOD BREAKDOWN ({total_calls} total calls):")
    for method, count in method_counts.most_common():
        pct = count / total_calls * 100
        safe_print(f"  {method:16s}: {count:3d}  ({pct:.1f}%)")
    safe_print(f"  → Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    failed_rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        for run in range(N_RUNS):
            if run_data[run]["parse_methods"][i] == "failed":
                failed_rows.append({
                    "lang": lang,
                    "q_idx": i + 1,
                    "run": run + 1,
                    "question": row["question"][:100],
                    "expected": row["answer"],
                    "cot_tail": "\n".join(run_data[run]["cots"][i].strip().split("\n")[-6:]),
                })

    if failed_rows:
        safe_print(f"\n[{lang.upper()}] ❌ FAILED PARSE DETAILS:")
        for fr in failed_rows:
            safe_print(f"  Q{fr['q_idx']} Run{fr['run']} | expected={fr['expected']}")
            safe_print(f"  Question: {fr['question']}")
            safe_print(f"  --- Last 6 lines of CoT ---")
            safe_print(fr["cot_tail"])
            safe_print(f"  {'─'*40}")
        failed_df = pd.DataFrame(failed_rows)
    else:
        failed_df = pd.DataFrame()
        safe_print(f"\n[{lang.upper()}] ✅ All calls parsed successfully!")

    total = len(df)
    safe_print(f"\n[{lang.upper()}] ACCURACY:")
    for run in range(N_RUNS):
        acc = sum(run_data[run]["corrects"])
        safe_print(f"  Run {run+1}: {acc}/{total} = {acc/total:.1%}")

    majority_acc = df["majority_correct"].sum()
    safe_print(f"  Majority vote: {majority_acc}/{total} = {majority_acc/total:.1%}")

    status_counts = df["vote_status"].value_counts()
    for status, count in status_counts.items():
        safe_print(f"  {status:12s}: {count} questions")

    # ── Individual per-language file (unchanged) ───────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, f"cot_majority_{lang}.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="parse_failures", index=False)
    safe_print(f"✅ Saved: {output_path}")

    return lang, majority_acc, total, failed_rows, df, failed_df


if __name__ == "__main__":
    summary = []
    all_failed_parses = []
    lang_results = {}   # lang -> {"df": ..., "failed_df": ...}

    with ThreadPoolExecutor(max_workers=LANG_WORKERS) as executor:
        future_to_lang = {
            executor.submit(run_inference_for_lang, lang): lang
            for lang in ALL_LANGUAGES
        }
        for future in as_completed(future_to_lang):
            lang, correct, total, failed_rows, df, failed_df = future.result()
            summary.append({
                "language": LANG_TO_FULL_NAME[lang],
                "code": lang,
                "correct": correct,
                "total": total,
                "accuracy": f"{correct/total:.1%}",
                "parse_failures": len(failed_rows),
            })
            all_failed_parses.extend(failed_rows)
            lang_results[lang] = {"df": df, "failed_df": failed_df}

    print("\n" + "="*60)
    print("SUMMARY (majority vote)")
    print("="*60)
    summary_df = pd.DataFrame(summary).sort_values("language")
    print(summary_df.to_string(index=False))

    total_failures = sum(r["parse_failures"] for r in summary)
    print(f"\nTotal parse failures across all languages: {total_failures}")

    # ── Individual summary file (unchanged) ───────────────────────────────────
    summary_path = os.path.join(OUTPUT_DIR, "summary_majority.xlsx")
    all_failed_df = pd.DataFrame(all_failed_parses) if all_failed_parses else pd.DataFrame()

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not all_failed_df.empty:
            all_failed_df.to_excel(writer, sheet_name="all_parse_failures", index=False)

    print(f"\n✅ Summary saved: {summary_path}")
    if not all_failed_df.empty:
        print(f"⚠️  Parse failure log also saved in 'all_parse_failures' sheet — inspect these CoTs to improve parsing!")

    # ── Combined final file (extra) ────────────────────────────────────────────
    # Sheet layout:
    #   summary            — accuracy + parse failure count across all languages
    #   results_{lang}     — full results for each language (one sheet each)
    #   failures_{lang}    — failed parse cases per language (only if any failures)
    # ──────────────────────────────────────────────────────────────────────────
    final_path = os.path.join(OUTPUT_DIR, "final_results.xlsx")
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for lang in sorted(lang_results.keys()):
            lang_results[lang]["df"].to_excel(writer, sheet_name=f"results_{lang}"[:31], index=False)
            if not lang_results[lang]["failed_df"].empty:
                lang_results[lang]["failed_df"].to_excel(writer, sheet_name=f"failures_{lang}"[:31], index=False)

    print(f"\n✅ Combined final file saved: {final_path}")
    if total_failures > 0:
        print(f"   ⚠️  failures_{{lang}} sheets present — inspect to improve parsing!")



