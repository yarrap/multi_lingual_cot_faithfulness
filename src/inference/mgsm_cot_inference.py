import os
import re
import cohere
import pandas as pd
from collections import Counter
from time import sleep
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mgsm

co = cohere.ClientV2(API_KEY)
OUTPUT_DIR = mgsm.OUTPUT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 3  # number of times to call the API per question


def parse_answer(cot_text: str, answer_prefix: str) -> str:
    lines = cot_text.strip().split("\n")
    
    # ── Primary: original logic ────────────────────────────────────────────────
    for line in reversed(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line:
            numbers = re.findall(r"\d+\.?\d*", clean_line.replace(",", ""))
            if numbers:
                return numbers[-1].rstrip(".")
    
    # ── Fallback 1: prefix is on its own line, number is on the NEXT line ──────
    # handles:  సమాధానం:        handles:  **సమాధానం:**
    #           \boxed{4}                  ప్రతి అక్వైరియంలో 14 చేపలు ఉన్నాయి.
    for i, line in enumerate(lines):
        clean_line = line.replace("**", "").strip()
        if answer_prefix in clean_line:
            # check if there's a next line
            if i + 1 < len(lines):
                next_line = lines[i + 1].replace("**", "").strip()
                # handle \boxed{276,000} or \boxed{4}
                boxed = re.findall(r'\\boxed\{([\d,]+)\}', next_line)
                if boxed:
                    return boxed[0].replace(",", "")
                # plain number on next line
                next_clean = next_line.replace("$", "").replace("₹", "").replace("%", "")
                numbers = re.findall(r"\d+\.?\d*", next_clean.replace(",", ""))
                if numbers:
                    return numbers[0].rstrip(".")
    
    # ── Fallback 2: \boxed{} anywhere in the last 5 lines ─────────────────────
    # handles cases where model uses LaTeX box without the prefix nearby
    for line in reversed(lines[-5:]):
        boxed = re.findall(r'\\boxed\{([\d,]+)\}', line)
        if boxed:
            return boxed[0].replace(",", "")
    
    return ""

# def parse_answer(cot_text: str, answer_prefix: str) -> str:
#     lines = cot_text.strip().split("\n")
#     for line in reversed(lines):
#         if answer_prefix in line:
#             numbers = re.findall(r"\d+\.?\d*", line.replace(",", ""))
#             if numbers:
#                 return numbers[-1].rstrip(".")
#     return ""


def answers_are_equal(extracted: str, expected: str) -> bool:
    try:
        return float(str(extracted).replace(",", "").strip()) == \
               float(str(expected).replace(",", "").strip())
    except (ValueError, TypeError):
        return False


def majority_vote(answers: list[str]) -> tuple[str, str]:
    """
    Return (voted_answer, vote_status) where vote_status is one of:
      'unanimous'   — all runs agree           e.g. [18, 18, 18]
      'majority'    — 2 out of 3 agree         e.g. [18, 18, 9]
      'all_differ'  — all 3 different          e.g. [120, 160, 175] → fallback to run 1
      'no_answer'   — all runs returned empty
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
        # all different — fall back to run 1's answer
        return valid[0], "all_differ"


def call_with_retry(co: cohere.ClientV2, prompt: str, model=MODEL_NAME, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024,
                temperature=0.0,
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
    print(f"Running MGSM {N_RUNS}-run majority vote for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    print(f"{'='*60}")

    df = pd.read_csv(mgsm.LANG_TO_DATA_PATH[lang])
    df = df[["question", "answer"]].copy()
    df = df.head(10)

    instruction_template = mgsm.LANG_TO_INSTRUCTIONS[lang]
    answer_prefix = mgsm.LANG_TO_ANSWER_PREFIX[lang]

    # Storage: one list per run
    all_cots       = [[] for _ in range(N_RUNS)]   # all_cots[run][question]
    all_extracted  = [[] for _ in range(N_RUNS)]   # all_extracted[run][question]

    for idx, row in df.iterrows():
        question = row["question"]
        print(f"\nQ{idx+1}: {question[:80]}...")
        prompt = instruction_template.format(input=question)

        run_answers = []
        for run in range(N_RUNS):
            try:
                cot = call_with_retry(co, prompt)
                extracted = parse_answer(cot, answer_prefix)
                print(f"  Run {run+1}: extracted={extracted!r:>10} | expected={row['answer']}")
                all_cots[run].append(cot)
                all_extracted[run].append(extracted)
                run_answers.append(extracted)
            except Exception as e:
                print(f"  Run {run+1} ERROR: {type(e).__name__}: {e}")
                all_cots[run].append("ERROR")
                all_extracted[run].append("")
                run_answers.append("")
            sleep(2)  # avoid rate limits between runs

        vote, status = majority_vote(run_answers)
        flag = " ⚠️  ALL DIFFER — falling back to run 1" if status == "all_differ" else ""
        print(f"  → [{status}] vote: {vote!r} | expected: {row['answer']}{flag}")
        sleep(1)

    # ── Build output DataFrame ─────────────────────────────────────────────────
    # Columns: question | answer
    #          cot_run1 | extracted_run1 | correct_run1
    #          cot_run2 | extracted_run2 | correct_run2
    #          cot_run3 | extracted_run3 | correct_run3
    #          majority_vote | majority_correct

    for run in range(N_RUNS):
        df[f"cot_run{run+1}"]        = all_cots[run]
        df[f"extracted_run{run+1}"]  = all_extracted[run]
        df[f"correct_run{run+1}"]    = df.apply(
            lambda r, run=run: answers_are_equal(all_extracted[run][r.name], r["answer"]),
            axis=1
        )

    vote_results = [
        majority_vote([all_extracted[run][i] for run in range(N_RUNS)])
        for i in range(len(df))
    ]
    df["majority_vote"]   = [v for v, _ in vote_results]
    df["vote_status"]     = [s for _, s in vote_results]  # unanimous / majority / all_differ / no_answer
    df["majority_correct"] = df.apply(
        lambda r: answers_are_equal(r["majority_vote"], r["answer"]), axis=1
    )

    # ── Per-run accuracies ─────────────────────────────────────────────────────
    total = len(df)
    for run in range(N_RUNS):
        acc = df[f"correct_run{run+1}"].sum()
        print(f"[{lang.upper()}] Run {run+1} accuracy: {acc}/{total} = {acc/total:.1%}")

    majority_acc = df["majority_correct"].sum()
    print(f"[{lang.upper()}] Majority vote accuracy: {majority_acc}/{total} = {majority_acc/total:.1%}")

    # show how often each vote status occurred
    status_counts = df["vote_status"].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:12s}: {count} questions")

    # ── Save to Excel ──────────────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, f"cot_majority_{lang}.xlsx")
    df.to_excel(output_path, index=False)
    print(f"✅ Saved: {output_path}")

    return lang, majority_acc, total


if __name__ == "__main__":
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

    print("\n" + "="*60)
    print("SUMMARY (majority vote)")
    print("="*60)
    summary_df = pd.DataFrame(summary)
    print(summary_df.to_string(index=False))

    summary_path = os.path.join(OUTPUT_DIR, "summary_majority.xlsx")
    summary_df.to_excel(summary_path, index=False)
    print(f"\n✅ Summary saved: {summary_path}")










# -----------------------------------------------------------------------------

# import os
# import re
# import cohere
# import pandas as pd
# from time import sleep
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mgsm

# co = cohere.ClientV2(API_KEY)
# OUTPUT_DIR = mgsm.OUTPUT_DIR
# os.makedirs(OUTPUT_DIR, exist_ok=True)


# # ── FIX 1: search on the LAST Answer: line, not split ─────────────────────────
# # Old code used cot_text.split(answer_prefix)[-1] which could grab a mid-CoT
# # occurrence. Searching line-by-line from the bottom is more reliable.
# def parse_answer(cot_text: str, answer_prefix: str) -> str:
#     lines = cot_text.strip().split("\n")
#     for line in reversed(lines):
#         if answer_prefix in line:
#             numbers = re.findall(r"\d+\.?\d*", line.replace(",", ""))
#             if numbers:
#                 return numbers[-1].rstrip(".")
#     return ""


# # ── FIX 2: float comparison instead of string comparison ──────────────────────
# # Old:  "18.0" == "18"  → False  ❌
# # New:  float("18.0") == float("18")  → True  ✅
# def answers_are_equal(extracted: str, expected: str) -> bool:
#     try:
#         return float(str(extracted).replace(",", "").strip()) == \
#                float(str(expected).replace(",", "").strip())
#     except (ValueError, TypeError):
#         return False


# def call_with_retry(co: cohere.ClientV2, prompt: str, model=MODEL_NAME, max_retries=5):
#     for attempt in range(max_retries):
#         try:
#             response = co.chat(
#                 model=model,
#                 messages=[{"role": "user", "content": prompt}],
#                 max_tokens=1024,   # FIX 3: was 500, bumped to avoid CoT truncation
#                 temperature=0.0,   # greedy decoding — correct for benchmarks
#                 # NOTE: top_p / top_k are irrelevant at temperature=0.0
#                 # (greedy decoding always picks the highest probability token)
#             )
#             return response.message.content[0].text.strip()
#         except Exception as e:
#             error_str = str(e).lower()
#             if "rate" in error_str or "limit" in error_str or "429" in error_str:
#                 wait = 60 * (attempt + 1)
#                 print(f"  Rate limit hit. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
#                 sleep(wait)
#             else:
#                 raise e
#     raise Exception(f"Failed after {max_retries} retries due to rate limiting")


# def run_inference_for_lang(lang: str):
#     print(f"\n{'='*60}")
#     print(f"Running MGSM inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
#     print(f"{'='*60}")

#     df = pd.read_csv(mgsm.LANG_TO_DATA_PATH[lang])
#     df = df[["question", "answer"]].copy()

#     instruction_template = mgsm.LANG_TO_INSTRUCTIONS[lang]
#     answer_prefix = mgsm.LANG_TO_ANSWER_PREFIX[lang]

#     cot_answers, extracted_answers = [], []

#     for idx, row in df.iterrows():
#         question = row["question"]
#         print(f"\nProcessing Q{idx+1}: {question[:80]}...")
#         prompt = instruction_template.format(input=question)

#         try:
#             cot = call_with_retry(co, prompt)
#             extracted = parse_answer(cot, answer_prefix)
#             print(f"Extracted: {extracted} | Expected: {row['answer']}")
#             cot_answers.append(cot)
#             extracted_answers.append(extracted)
#         except Exception as e:
#             print(f"Error on Q{idx+1}: {type(e).__name__}: {e}")
#             cot_answers.append("ERROR")
#             extracted_answers.append("")

#         sleep(2)

#     df["cot_answer"] = cot_answers
#     df["extracted_answer"] = extracted_answers
#     df["is_correct"] = df.apply(
#         lambda r: answers_are_equal(r["extracted_answer"], r["answer"]), axis=1
#     )

#     accuracy = df["is_correct"].sum()
#     total = len(df)
#     print(f"\n[{lang.upper()}] Accuracy: {accuracy}/{total} = {accuracy/total:.1%}")

#     output_path = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
#     df.to_csv(output_path, index=False, quoting=1)
#     print(f"✅ Saved: {output_path}")

#     return lang, accuracy, total


# if __name__ == "__main__":
#     summary = []
#     for lang in ALL_LANGUAGES:
#         lang, correct, total = run_inference_for_lang(lang)
#         summary.append({
#             "language": LANG_TO_FULL_NAME[lang],
#             "code": lang,
#             "correct": correct,
#             "total": total,
#             "accuracy": f"{correct/total:.1%}"
#         })
#         sleep(5)

#     print("\n" + "="*60)
#     print("SUMMARY")
#     print("="*60)
#     summary_df = pd.DataFrame(summary)
#     print(summary_df.to_string(index=False))
#     summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
#     summary_df.to_csv(summary_path, index=False)
#     print(f"\n✅ Summary saved: {summary_path}")








# # -------------------------------------------------------------------------------

# # import os
# # import re
# # import cohere
# # import pandas as pd
# # from time import sleep
# # import sys
# # import os
# # sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# # from configs import API_KEY, MODEL_NAME, ALL_LANGUAGES, LANG_TO_FULL_NAME, mgsm

# # co = cohere.ClientV2(API_KEY)
# # OUTPUT_DIR = mgsm.OUTPUT_DIR
# # os.makedirs(OUTPUT_DIR, exist_ok=True)


# # def parse_answer(cot_text: str, answer_prefix: str) -> str:
# #     if answer_prefix not in cot_text:
# #         return ""
# #     answer_text = cot_text.split(answer_prefix)[-1].strip()
# #     numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))
# #     return numbers[-1].rstrip(".") if numbers else ""


# # def call_with_retry(co: cohere.ClientV2, prompt: str, model=MODEL_NAME, max_retries=5):
# #     for attempt in range(max_retries):
# #         try:
# #             response = co.chat(
# #                 model=model,
# #                 messages=[{"role": "user", "content": prompt}],
# #                 max_tokens=500,
# #                 temperature=0.0,
# #             )
# #             return response.message.content[0].text.strip()
# #         except Exception as e:
# #             error_str = str(e).lower()
# #             if "rate" in error_str or "limit" in error_str or "429" in error_str:
# #                 wait = 60 * (attempt + 1)
# #                 print(f"  Rate limit hit. Waiting {wait}s before retry {attempt+1}/{max_retries}...")
# #                 sleep(wait)
# #             else:
# #                 raise e
# #     raise Exception(f"Failed after {max_retries} retries due to rate limiting")


# # def run_inference_for_lang(lang: str):
# #     print(f"\n{'='*60}")
# #     print(f"Running MGSM inference for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
# #     print(f"{'='*60}")

# #     df = pd.read_csv(mgsm.LANG_TO_DATA_PATH[lang])
# #     df = df[["question", "answer"]].copy()

# #     instruction_template = mgsm.LANG_TO_INSTRUCTIONS[lang]
# #     answer_prefix = mgsm.LANG_TO_ANSWER_PREFIX[lang]

# #     cot_answers, extracted_answers = [], []

# #     for idx, row in df.iterrows():
# #         question = row["question"]
# #         print(f"\nProcessing Q{idx+1}: {question[:80]}...")
# #         prompt = instruction_template.format(input=question)

# #         try:
# #             cot = call_with_retry(co, prompt)
# #             extracted = parse_answer(cot, answer_prefix)
# #             print(f"Extracted: {extracted} | Expected: {row['answer']}")
# #             cot_answers.append(cot)
# #             extracted_answers.append(extracted)
# #         except Exception as e:
# #             print(f"Error on Q{idx+1}: {type(e).__name__}: {e}")
# #             cot_answers.append("ERROR")
# #             extracted_answers.append("")

# #         sleep(2)

# #     df["cot_answer"] = cot_answers
# #     df["extracted_answer"] = extracted_answers
# #     df["is_correct"] = df.apply(
# #         lambda r: str(r["extracted_answer"]).strip() == str(r["answer"]).strip(), axis=1
# #     )

# #     accuracy = df["is_correct"].sum()
# #     total = len(df)
# #     print(f"\n[{lang.upper()}] Accuracy: {accuracy}/{total} = {accuracy/total:.1%}")

# #     output_path = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
# #     df.to_csv(output_path, index=False, quoting=1)
# #     print(f"✅ Saved: {output_path}")

# #     return lang, accuracy, total


# # if __name__ == "__main__":
# #     summary = []
# #     # Only zh by default as in previous script? No, let's use all from config if appropriate
# #     # The previous script had ALL_LANGUAGES = ["zh"] which was strange. 
# #     # I'll stick to the config ALL_LANGUAGES but maybe only zh was requested before.
# #     # Actually, I'll use ALL_LANGUAGES but I'll check if the user had a specific reason for zh only.
# #     for lang in ALL_LANGUAGES: # Keeping it as per the specifically previous version but using config for others if needed
# #         lang, correct, total = run_inference_for_lang(lang)
# #         summary.append({
# #             "language": LANG_TO_FULL_NAME[lang],
# #             "code": lang,
# #             "correct": correct,
# #             "total": total,
# #             "accuracy": f"{correct/total:.1%}"
# #         })
# #         sleep(5)
