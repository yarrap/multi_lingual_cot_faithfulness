import os
import re
import cohere
import pandas as pd
from collections import Counter
from time import sleep
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mmlu

co = cohere.ClientV2(API_KEY)
OUTPUT_DIR = mmlu.OUTPUT_DIR.replace("cot_inference", "direct_inference")  # Change output dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 3
MAX_WORKERS = 3
LANG_WORKERS = 1

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)



def parse_answer(response_text: str, answer_prefix: str, lang: str = "") -> tuple[str, str]:
    """
    Extract the predicted letter (A/B/C/D) from direct answer.
    For direct prompting, answer should be just a letter.
    Returns: (extracted_answer, parse_method)
    """
    response_text = response_text.strip()
    
    # Method 1: Response is just a single letter
    if re.match(r'^[A-Da-d]$', response_text):
        return response_text.upper(), "single_letter"
    
    # Method 2: Response is letter with period or parentheses
    match = re.match(r'^[(\[]?([A-Da-d])[)\].]?\s*$', response_text)
    if match:
        return match.group(1).upper(), "letter_with_punct"
    
    # Method 3: First letter in response
    first_line = response_text.split('\n')[0].strip()
    match = re.match(r'^[(\[]?([A-Da-d])[)\].]?', first_line)
    if match:
        return match.group(1).upper(), "first_letter"
    
    # Method 4: Any standalone A/B/C/D
    matches = re.findall(r'\b([A-D])\b', response_text.upper())
    if matches:
        return matches[0], "first_occurrence"  # Take FIRST not last for direct
    
    # Method 5: Scan for any letter
    for letter in ['A', 'B', 'C', 'D']:
        if letter in response_text.upper():
            return letter, "letter_scan"
    
    return "", "failed"


def answers_are_equal(extracted: str, expected: str) -> bool:
    """Check if extracted answer matches expected answer."""
    return str(extracted).strip().upper() == str(expected).strip().upper()


def majority_vote(answers: list[str]) -> tuple[str, str]:
    """
    Determine majority vote from multiple answers.
    Returns: (voted_answer, status)
    """
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
    """Call Cohere API with retry logic for rate limits."""
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,  # Direct answer needs very few tokens
                temperature=0.0,
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


def fetch_single_run(run_idx: int, prompt: str, answer_prefix: str, 
                     expected: str, q_label: str, lang: str = "", debug: bool = False):
    """Execute a single inference run with optional debugging."""
    try:
        response = call_with_retry(prompt)
        extracted, method = parse_answer(response, answer_prefix, lang)
        correct = answers_are_equal(extracted, expected)
        
        # Debug mode: show response for incorrect answers
        if debug and not correct:
            safe_print(f"\n  [DEBUG] {q_label} Run {run_idx+1} - INCORRECT:")
            safe_print(f"  Expected: {expected}, Got: {extracted}, Method: {method}")
            safe_print(f"  Raw response: {response!r}")
            safe_print(f"  {'-'*50}")
        
        safe_print(
            f"  {q_label} Run {run_idx+1}: "
            f"extracted={extracted!r:>3} | expected={expected} | "
            f"method={method:20s} | {'✓' if correct else '✗'}"
        )
        return {
            "run": run_idx,
            "response": response,
            "extracted": extracted,
            "parse_method": method,
            "correct": correct,
            "error": None,
        }
    except Exception as e:
        safe_print(f"  {q_label} Run {run_idx+1} ERROR: {type(e).__name__}: {e}")
        return {
            "run": run_idx,
            "response": "ERROR",
            "extracted": "",
            "parse_method": "error",
            "correct": False,
            "error": str(e),
        }


def run_inference_for_lang(lang: str, debug: bool = False):
    """Run 3-run majority vote inference for a single language."""
    safe_print(f"\n{'='*60}")
    safe_print(f"Running MMLU {N_RUNS}-run DIRECT (no CoT) for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    safe_print(f"{'='*60}")

    df = pd.read_csv(mmlu.LANG_TO_DATA_PATH[lang])

    MATH_SUBJECTS = [
        'abstract_algebra',
        'college_mathematics',
        'elementary_mathematics',
        'high_school_mathematics',
        'high_school_statistics'
    ]
    
    if 'subject' in df.columns:
        original_count = len(df)
        df = df[df['subject'].isin(MATH_SUBJECTS)].copy()
        safe_print(f"Filtered from {original_count} to {len(df)} math questions")
        safe_print(f"Math subjects: {', '.join(MATH_SUBJECTS)}\n")
    else:
        safe_print("⚠️  Warning: 'subject' column not found - processing all questions\n")
    
    df = df[["question", "A", "B", "C", "D", "answer"]].copy()
    df = df.reset_index(drop=True)  # Reset index for sequential numbering
    # df = df.head(10)  # Uncomment for testing

    instruction_template = mmlu.LANG_TO_DIRECT_INSTRUCTIONS[lang]
    answer_prefix = mmlu.LANG_TO_DIRECT_ANSWER_PREFIX[lang]

    run_data = {run: {"responses": [], "extracted": [], "parse_methods": [], "corrects": []}
                for run in range(N_RUNS)}

    for idx, row in df.iterrows():
        question = row["question"]
        q_label = f"[{lang.upper()}] Q{idx+1}"
        safe_print(f"\n{q_label}: {question[:80]}...")
        prompt = instruction_template.format(
            question=question,
            A=row["A"],
            B=row["B"],
            C=row["C"],
            D=row["D"],
        )

        # Parallel execution of 3 runs
        run_results = [None] * N_RUNS
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_run = {
                executor.submit(
                    fetch_single_run, run, prompt, answer_prefix, 
                    str(row["answer"]), q_label, lang, debug
                ): run
                for run in range(N_RUNS)
            }
            for future in as_completed(future_to_run):
                result = future.result()
                run_results[result["run"]] = result

        # Store results
        for run in range(N_RUNS):
            r = run_results[run]
            run_data[run]["responses"].append(r["response"])
            run_data[run]["extracted"].append(r["extracted"])
            run_data[run]["parse_methods"].append(r["parse_method"])
            run_data[run]["corrects"].append(r["correct"])

        # Compute majority vote
        answers_this_q = [run_results[run]["extracted"] for run in range(N_RUNS)]
        vote, status = majority_vote(answers_this_q)
        flag = " ⚠️  ALL DIFFER" if status == "all_differ" else ""
        safe_print(f"  → [{status}] vote={vote!r} | expected={row['answer']}{flag}")

    # Add run-specific columns to dataframe
    for run in range(N_RUNS):
        df[f"response_run{run+1}"]      = run_data[run]["responses"]
        df[f"extracted_run{run+1}"]     = run_data[run]["extracted"]
        df[f"parse_method_run{run+1}"]  = run_data[run]["parse_methods"]
        df[f"correct_run{run+1}"]       = run_data[run]["corrects"]

    # Compute majority vote for all questions
    vote_results = [
        majority_vote([run_data[run]["extracted"][i] for run in range(N_RUNS)])
        for i in range(len(df))
    ]
    df["majority_vote"]    = [v for v, _ in vote_results]
    df["vote_status"]      = [s for _, s in vote_results]
    df["majority_correct"] = df.apply(
        lambda r: answers_are_equal(r["majority_vote"], r["answer"]), axis=1
    )

    # Parse method breakdown
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
        safe_print(f"  {method:25s}: {count:3d}  ({pct:.1f}%)")
    safe_print(f"  → Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    # Collect failed parses
    failed_rows = []
    for i, row in df.iterrows():
        for run in range(N_RUNS):
            if run_data[run]["parse_methods"][i] == "failed":
                failed_rows.append({
                    "lang": lang,
                    "q_idx": i + 1,
                    "run": run + 1,
                    "question": row["question"][:100],
                    "expected": row["answer"],
                    "response": run_data[run]["responses"][i],
                })

    if failed_rows:
        safe_print(f"\n[{lang.upper()}] ❌ FAILED PARSE DETAILS:")
        for fr in failed_rows:
            safe_print(f"  Q{fr['q_idx']} Run{fr['run']} | expected={fr['expected']}")
            safe_print(f"  Question: {fr['question']}")
            safe_print(f"  Response: {fr['response']!r}")
            safe_print(f"  {'─'*40}")
        failed_df = pd.DataFrame(failed_rows)
    else:
        failed_df = pd.DataFrame()
        safe_print(f"\n[{lang.upper()}] ✅ All calls parsed successfully!")

    # Accuracy summary
    total = len(df)
    safe_print(f"\n[{lang.upper()}] ACCURACY (DIRECT - NO COT):")
    for run in range(N_RUNS):
        acc = sum(run_data[run]["corrects"])
        safe_print(f"  Run {run+1}: {acc}/{total} = {acc/total:.1%}")

    majority_acc = df["majority_correct"].sum()
    safe_print(f"  Majority vote: {majority_acc}/{total} = {majority_acc/total:.1%}")

    status_counts = df["vote_status"].value_counts()
    for status, count in status_counts.items():
        safe_print(f"  {status:12s}: {count} questions")

    # Save individual language file
    output_path = os.path.join(OUTPUT_DIR, f"direct_majority_{lang}.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="parse_failures", index=False)
    safe_print(f"✅ Saved: {output_path}")

    return lang, majority_acc, total, failed_rows, df, failed_df


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    args = parser.parse_args()
    
    summary = []
    all_failed_parses = []
    lang_results = {}

    # Sequential language execution
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
        lang_results[lang] = {"df": df, "failed_df": failed_df}

    print("\n" + "="*60)
    print("SUMMARY - DIRECT PROMPTING (NO COT)")
    print("="*60)
    summary_df = pd.DataFrame(summary).sort_values("language")
    print(summary_df.to_string(index=False))

    total_failures = sum(r["parse_failures"] for r in summary)
    print(f"\nTotal parse failures across all languages: {total_failures}")

    # Save summary file
    summary_path = os.path.join(OUTPUT_DIR, "summary_direct.xlsx")
    all_failed_df = pd.DataFrame(all_failed_parses) if all_failed_parses else pd.DataFrame()

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not all_failed_df.empty:
            all_failed_df.to_excel(writer, sheet_name="all_parse_failures", index=False)

    print(f"\n✅ Summary saved: {summary_path}")

    # Save combined final file
    final_path = os.path.join(OUTPUT_DIR, "final_results_direct.xlsx")
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for lang in sorted(lang_results.keys()):
            lang_results[lang]["df"].to_excel(writer, sheet_name=f"results_{lang}"[:31], index=False)
            if not lang_results[lang]["failed_df"].empty:
                lang_results[lang]["failed_df"].to_excel(writer, sheet_name=f"failures_{lang}"[:31], index=False)

    print(f"\n✅ Combined final file saved: {final_path}")