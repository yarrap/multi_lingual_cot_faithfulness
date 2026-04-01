import os
import re
import cohere
import pandas as pd
from dotenv import load_dotenv
from collections import Counter
from time import sleep
import threading
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

load_dotenv()

api_key = os.getenv("COHERE_API_KEY")
if not api_key:
    raise ValueError("COHERE_API_KEY not found in .env file")

co = cohere.ClientV2(api_key)

ALL_LANGUAGES = ["sw", "te", "zh","en", "bn"]

model_name = "tiny-aya-global"

LANG_TO_PATH = {
    "en": "../../../../datasets/mgsm_en.csv",
    "bn": "../../../../datasets/mgsm_bn.csv",
    "sw": "../../../../datasets/mgsm_sw.csv",
    "te": "../../../../datasets/mgsm_te.csv",
    "zh": "../../../../datasets/mgsm_zh.csv",
}

# ── CHANGE 1: Instruction retained + 2 format examples added ──────────────────
# The original instruction line is kept exactly as it was.
# Two simple examples are added below it so the model sees the exact
# format expected: question → bare number via the answer prefix.
# Examples are trivial on purpose — the model should focus on FORMAT not math.
# ──────────────────────────────────────────────────────────────────────────────
LANG_TO_INSTRUCTIONS = {
    "en": """Answer the following math question with a single number only. No explanation.

Question: Tom has 5 apples. He eats 2. How many are left?
Answer: 3

Question: A shop has 10 items. 4 are sold. How many remain?
Answer: 6

Question: {input}
Answer:""",

    "bn": """নিচের গণিত প্রশ্নের উত্তর শুধুমাত্র একটি সংখ্যায় দিন। কোনো ব্যাখ্যা নয়।

প্রশ্ন: টমের কাছে ৫টি আপেল আছে। সে ২টি খায়। কতটি বাকি আছে?
উত্তর: 3

প্রশ্ন: একটি দোকানে ১০টি জিনিস আছে। ৪টি বিক্রি হয়। কতটি বাকি?
উত্তর: 6

প্রশ্ন: {input}
উত্তর:""",

    "sw": """Jibu swali hili la hesabu kwa nambari moja tu. Bila maelezo.

Swali: Tomi ana maapulo 5. Anakula 2. Amebaki na maapulo mangapi?
Jibu: 3

Swali: Duka lina vitu 10. Vitu 4 vinauzwa. Vimebaki vingapi?
Jibu: 6

Swali: {input}
Jibu:""",

    "te": """క్రింది గణిత ప్రశ్నకు కేవలం ఒక్క సంఖ్యలో సమాధానం ఇవ్వండి. వివరణ అవసరం లేదు.

ప్రశ్న: టామ్ దగ్గర 5 ఆపిల్‌లు ఉన్నాయి. అతను 2 తింటాడు. ఎన్ని మిగిలాయి?
సమాధానం: 3

ప్రశ్న: ఒక దుకాణంలో 10 వస్తువులు ఉన్నాయి. 4 అమ్ముడయ్యాయి. ఎన్ని మిగిలాయి?
సమాధానం: 6

ప్రశ్న: {input}
సమాధానం:""",

    "zh": """请用一个数字回答以下数学题。不需要解释。

问题：汤姆有5个苹果。他吃了2个。还剩几个？
答案：3

问题：一家商店有10件商品。卖出了4件。还剩几件？
答案：6

问题：{input}
答案：""",
}

# Answer prefixes for the 3-stage parser — unchanged
LANG_TO_ANSWER_PREFIX = {
    "en": ["Answer:"],
    "bn": ["উত্তর:"],
    "sw": ["Jibu:"],
    "te": ["సమాధానం:"],
    "zh": ["答案：", "答案:"],
}

LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}

OUTPUT_DIR = f"../../../../results/direct_inference/mgsm/{model_name}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

N_RUNS = 3
MAX_WORKERS = 3
LANG_WORKERS = 1

print_lock = threading.Lock()


def safe_print(*args, **kwargs):
    with print_lock:
        print(*args, **kwargs)


# ── CHANGE 2: Native numeral normalization ─────────────────────────────────────
# Covers Bengali (০-৯), Telugu (౦-౯), and Chinese written-form (〇-九).
# Swahili and English use ASCII digits already — no entry needed for them.
# Devanagari intentionally excluded as it is not a target language.
# Built into a translation table once at load time for efficiency.
# ──────────────────────────────────────────────────────────────────────────────
NATIVE_NUMERAL_MAP = {
    # Bengali digits
    "০": "0", "১": "1", "২": "2", "৩": "3", "৪": "4",
    "৫": "5", "৬": "6", "৭": "7", "৮": "8", "৯": "9",
    # Telugu digits
    "౦": "0", "౧": "1", "౨": "2", "౩": "3", "౪": "4",
    "౫": "5", "౬": "6", "౭": "7", "౮": "8", "౯": "9",
    # Chinese written-form numerals (edge case — model rarely uses these)
    "〇": "0", "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8", "九": "9",
}

_NUMERAL_TABLE = str.maketrans(NATIVE_NUMERAL_MAP)


def normalize_numerals(text: str) -> str:
    """Convert native-script digits to ASCII before parsing."""
    return text.translate(_NUMERAL_TABLE)


def parse_answer(response_text: str, lang: str) -> tuple[str, str]:
    """
    3-stage parser — identical logic to original, with normalize_numerals()
    added at the top so all stages operate on ASCII digits only.

      1. Entire response is a bare number
      2. Native answer prefix found e.g. "Answer: 42"
      3. Fallback — grab the last number anywhere in the response
    """
    if not response_text or response_text.strip() == "":
        return "", "failed"

    # Normalise native numerals before any regex work
    normalised = normalize_numerals(response_text)
    text = normalised.replace(",", "").strip()

    # Stage 1: pure number
    if re.fullmatch(r"\d+\.?\d*", text):
        return text.rstrip("."), "bare_number"

    # Stage 2: native prefix found
    for prefix in LANG_TO_ANSWER_PREFIX.get(lang, []):
        if prefix in normalised:
            after = normalised.split(prefix)[-1].strip()
            numbers = re.findall(r"\d+\.?\d*", after.replace(",", ""))
            if numbers:
                return numbers[0].rstrip("."), "prefix_match"

    # Stage 3: last number fallback
    numbers = re.findall(r"\d+\.?\d*", text)
    if numbers:
        return numbers[-1].rstrip("."), "last_number"

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


def call_with_retry(prompt: str, model=model_name, max_retries=5) -> str:
    for attempt in range(max_retries):
        try:
            response = co.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.3,
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


def fetch_single_run(run_idx: int, prompt: str, expected: str, q_label: str, lang: str):
    try:
        raw = call_with_retry(prompt)
        extracted, method = parse_answer(raw, lang)
        correct = answers_are_equal(extracted, expected)
        safe_print(
            f"  {q_label} Run {run_idx+1}: "
            f"raw={raw!r:>15} | extracted={extracted!r:>10} | "
            f"expected={expected} | method={method} | {'✓' if correct else '✗'}"
        )
        return {
            "run": run_idx,
            "raw": raw,
            "extracted": extracted,
            "parse_method": method,
            "correct": correct,
            "error": None,
        }
    except Exception as e:
        safe_print(f"  {q_label} Run {run_idx+1} ERROR: {type(e).__name__}: {e}")
        return {
            "run": run_idx,
            "raw": "ERROR",
            "extracted": "",
            "parse_method": "error",
            "correct": False,
            "error": str(e),
        }


def run_inference_for_lang(lang: str):
    safe_print(f"\n{'='*60}")
    safe_print(f"Running DIRECT {N_RUNS}-run majority vote for: {LANG_TO_FULL_NAME[lang]} ({lang.upper()})")
    safe_print(f"{'='*60}")

    df = pd.read_csv(LANG_TO_PATH[lang])
    df = df[["question", "answer"]].copy()
    # df = df.head(10)

    instruction_template = LANG_TO_INSTRUCTIONS[lang]

    run_data = {run: {"raws": [], "extracted": [], "parse_methods": [], "corrects": []}
                for run in range(N_RUNS)}

    for idx, row in df.iterrows():
        question = row["question"]
        q_label = f"[{lang.upper()}] Q{idx+1}"
        safe_print(f"\n{q_label}: {question[:80]}...")
        prompt = instruction_template.format(input=question)

        run_results = [None] * N_RUNS
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_run = {
                executor.submit(
                    fetch_single_run, run, prompt, str(row["answer"]), q_label, lang
                ): run
                for run in range(N_RUNS)
            }
            for future in as_completed(future_to_run):
                result = future.result()
                run_results[result["run"]] = result

        for run in range(N_RUNS):
            r = run_results[run]
            run_data[run]["raws"].append(r["raw"])
            run_data[run]["extracted"].append(r["extracted"])
            run_data[run]["parse_methods"].append(r["parse_method"])
            run_data[run]["corrects"].append(r["correct"])

        answers_this_q = [run_results[run]["extracted"] for run in range(N_RUNS)]
        vote, status = majority_vote(answers_this_q)
        flag = " ⚠️  ALL DIFFER — falling back to run 1" if status == "all_differ" else ""
        safe_print(f"  → [{status}] vote={vote!r} | expected={row['answer']}{flag}")

    # ── Build output DataFrame ─────────────────────────────────────────────────
    for run in range(N_RUNS):
        df[f"raw_run{run+1}"]          = run_data[run]["raws"]
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

    # ── Parse method breakdown ─────────────────────────────────────────────────
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

    # ── Collect failed parse rows ──────────────────────────────────────────────
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
                    "raw_response": run_data[run]["raws"][i],
                })

    if failed_rows:
        safe_print(f"\n[{lang.upper()}] ❌ FAILED PARSE DETAILS:")
        for fr in failed_rows:
            safe_print(f"  Q{fr['q_idx']} Run{fr['run']} | expected={fr['expected']}")
            safe_print(f"  Question: {fr['question']}")
            safe_print(f"  Raw response: {fr['raw_response']!r}")
            safe_print(f"  {'─'*40}")
        failed_df = pd.DataFrame(failed_rows)
    else:
        failed_df = pd.DataFrame()
        safe_print(f"\n[{lang.upper()}] ✅ All calls parsed successfully!")

    # ── Accuracy report ────────────────────────────────────────────────────────
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

    # ── Save per-language .xlsx ────────────────────────────────────────────────
    output_path = os.path.join(OUTPUT_DIR, f"direct_majority_{lang}.xlsx")
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="results", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="parse_failures", index=False)
    safe_print(f"✅ Saved: {output_path}")

    return lang, majority_acc, total, failed_rows, df, failed_df


# ── Run all languages ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary = []
    all_failed_parses = []
    lang_results = {}

    with ThreadPoolExecutor(max_workers=LANG_WORKERS) as executor:
        future_to_lang = {
            executor.submit(run_inference_for_lang, lang): lang
            for lang in ALL_LANGUAGES
        }
        for future in as_completed(future_to_lang):
            lang, correct, total, failed_rows, df, failed_df = future.result()
            summary.append({
                "language":       LANG_TO_FULL_NAME[lang],
                "code":           lang,
                "correct":        correct,
                "total":          total,
                "accuracy":       f"{correct/total:.1%}",
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

    summary_path = os.path.join(OUTPUT_DIR, "summary_majority.xlsx")
    all_failed_df = pd.DataFrame(all_failed_parses) if all_failed_parses else pd.DataFrame()

    with pd.ExcelWriter(summary_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        if not all_failed_df.empty:
            all_failed_df.to_excel(writer, sheet_name="all_parse_failures", index=False)

    print(f"\n✅ Summary saved: {summary_path}")
    if not all_failed_df.empty:
        print(f"⚠️  Parse failure log also saved in 'all_parse_failures' sheet.")

    final_path = os.path.join(OUTPUT_DIR, "final_results.xlsx")
    with pd.ExcelWriter(final_path, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="summary", index=False)
        for lang in sorted(lang_results.keys()):
            lang_results[lang]["df"].to_excel(
                writer, sheet_name=f"results_{lang}"[:31], index=False)
            if not lang_results[lang]["failed_df"].empty:
                lang_results[lang]["failed_df"].to_excel(
                    writer, sheet_name=f"failures_{lang}"[:31], index=False)

    print(f"\n✅ Combined final file saved: {final_path}")
    if total_failures > 0:
        print(f"   ⚠️  failures_{{lang}} sheets present — inspect to improve parsing!")