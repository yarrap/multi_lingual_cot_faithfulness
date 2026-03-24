"""
gemma3_mgsm_vllm_direct.py
===========================
MGSM direct (no CoT) evaluation for Gemma-3-4B-IT using vLLM + 3-run majority voting.

Expected input files:
    /content/mgsm_en.csv   (columns: question, answer)
    /content/mgsm_bn.csv   ... etc.

Output:
    /content/results/gemma3-4b-vllm-direct/direct_<lang>.csv
    /content/results/gemma3-4b-vllm-direct/summary.csv
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import re
import csv
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────────────────────────
# CONFIG  ← edit to match your setup
# ─────────────────────────────────────────────────────────────────
HF_MODEL_ID  = "google/gemma-3-4b-it"
MODEL_NAME   = "gemma3-4b-vllm-direct"
OUTPUT_DIR   = f"/content/results/{MODEL_NAME}"


# Uncomment to run all languages:
ALL_LANGUAGES = ["en", "sw", "te", "zh", "bn"]

N_RUNS       = 3      # majority vote across this many independent generations
MAX_TOKENS   = 50
TEMPERATURE  = 0.0    # greedy — deterministic (set >0 for diverse runs)
TOP_P        = 0.95
GPU_MEM_UTIL = 0.90

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────────────────────────
# STATIC LOOKUP TABLES
# ─────────────────────────────────────────────────────────────────
LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}

LANG_TO_DATA_PATH = {
    "en": "/content/mgsm_en.csv",
    "bn": "/content/mgsm_bn.csv",
    "sw": "/content/mgsm_sw.csv",
    "te": "/content/mgsm_te.csv",
    "zh": "/content/mgsm_zh.csv",
}

LANG_TO_INSTRUCTIONS = {
    "en": """Answer the following math question with a single number only. No explanation.

Question: {input}
Answer:""",

    "bn": """নিচের গণিত প্রশ্নের উত্তর শুধুমাত্র একটি সংখ্যায় দিন। কোনো ব্যাখ্যা নয়।

প্রশ্ন: {input}
উত্তর:""",

    "sw": """Jibu swali hili la hesabu kwa nambari moja tu. Bila maelezo.

Swali: {input}
Jibu:""",

    "te": """క్రింది గణిత ప్రశ్నకు కేవలం ఒక్క సంఖ్యలో సమాధానం ఇవ్వండి. వివరణ అవసరం లేదు.

ప్రశ్న: {input}
సమాధానం:""",

    "zh": """请用一个数字回答以下数学题。不需要解释。

问题：{input}
答案：""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer:",
    "bn": "উত্তর:",
    "sw": "Jibu:",
    "te": "సమాధానం:",
    "zh": "答案：",
}

REQUIRED_COLS = ["question", "answer"]

# Build interleaved output columns (same layout as Qwen MMLU script)
_ITER_COLS: list[str] = []
for _r in range(N_RUNS):
    _ITER_COLS += [
        f"cot_run{_r+1}",
        f"extracted_run{_r+1}",
        f"parse_method_run{_r+1}",
        f"correct_run{_r+1}",
    ]

CKPT_COLS = (
    REQUIRED_COLS
    + _ITER_COLS
    + ["majority_vote", "vote_status", "majority_correct"]
)

# ─────────────────────────────────────────────────────────────────
# ANSWER PARSER  (full multi-stage — ported from original script)
# ─────────────────────────────────────────────────────────────────

def parse_answer(cot_text: str, answer_prefix: str, lang: str = "") -> tuple[str, str]:
    """
    Extract the numeric answer from a CoT string.
    Returns (answer_str, parse_method).
    """
    if not isinstance(cot_text, str):
        return "", "failed"

    lines = cot_text.strip().split("\n")

    def extract_numbers(text: str) -> list[str]:
        return re.findall(r"\d+\.?\d*", text.replace(",", ""))

    # ── Chinese-specific pre-pass ──────────────────────────────────────────
    if lang == "zh":
        def normalize_colons(s: str) -> str:
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

        def extract_boxed_zh(line: str) -> str:
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

    # ── Default — all languages except zh ─────────────────────────────────

    def extract_boxed(line: str) -> str:
        m = re.search(r'\\boxed\{([^}]+)\}', line)
        if not m:
            return ""
        inner = m.group(1)
        if "=" in inner:
            inner = inner.split("=")[-1]
        nums = re.findall(r"\d+\.?\d*", inner.replace(",", ""))
        return nums[-1].rstrip(".") if nums else ""

    def extract_inline_latex(line: str) -> str:
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

    # Fallback 1a: prefix on its own line → \\boxed{} on next line
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

    # Fallback 2: \\boxed{} anywhere in last 5 lines
    for line in reversed(lines[-5:]):
        val = extract_boxed(line)
        if val:
            return val, "fallback_box"

    # Fallback 3: inline LaTeX \\( num \\) in last 5 lines
    for line in reversed(lines[-5:]):
        val = extract_inline_latex(line)
        if val:
            return val, "fallback_inline_latex"

    # Fallback 4: bold answer pattern **number** in last 5 lines
    for line in reversed(lines[-5:]):
        bold_nums = re.findall(r'\*\*[^*]*?(\d+\.?\d*)[^*]*?\*\*', line.replace(",", ""))
        if bold_nums:
            return bold_nums[-1].rstrip("."), "fallback_bold"

    # Fallback 5: last number in entire CoT
    all_nums = re.findall(r"\d+\.?\d*", cot_text.replace(",", ""))
    if all_nums:
        return all_nums[-1].rstrip("."), "last_number"

    return "", "failed"


# ─────────────────────────────────────────────────────────────────
# VOTING HELPERS  (ported from Qwen3 MMLU script)
# ─────────────────────────────────────────────────────────────────

def answers_are_equal(extracted: str, expected: str) -> bool:
    try:
        return (
            float(str(extracted).replace(",", "").strip())
            == float(str(expected).replace(",", "").strip())
        )
    except (ValueError, TypeError):
        return False


def majority_vote(answers: list[str]) -> tuple[str, str]:
    """
    Return (most_common_answer, vote_status).
    status: 'unanimous' | 'majority' | 'all_differ' | 'no_answer'
    Falls back to run-1 answer when all differ.
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
        return valid[0], "all_differ"   # fall back to run-1


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL  (once, shared across all languages)
# ─────────────────────────────────────────────────────────────────
print(f"\nLoading tokenizer : {HF_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_ID)

print(f"Loading vLLM engine: {HF_MODEL_ID}")
llm = LLM(
    model=HF_MODEL_ID,
    trust_remote_code=True,
    gpu_memory_utilization=GPU_MEM_UTIL,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
)
print("Model loaded.\n")


# ─────────────────────────────────────────────────────────────────
# PER-LANGUAGE INFERENCE
# ─────────────────────────────────────────────────────────────────

def run_inference_for_lang(lang: str) -> dict:
    full_name     = LANG_TO_FULL_NAME[lang]
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]
    output_path   = os.path.join(OUTPUT_DIR, f"direct_{lang}.csv")
    ckpt_path     = os.path.join(OUTPUT_DIR, f"direct_{lang}_checkpoint.csv")

    # ── Skip if already complete ──────────────────────────────────────────
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) > 0:
            print(f"[{lang.upper()}] Already complete ({len(existing)} rows) — skipping.")
            majority_acc = int(existing["majority_correct"].sum())
            total        = len(existing)
            return {
                "language":       full_name,
                "code":           lang,
                "correct":        majority_acc,
                "total":          total,
                "accuracy":       f"{majority_acc/total:.1%}",
                "parse_failures": 0,
                "output_file":    output_path,
            }

    # ── Load & validate ───────────────────────────────────────────────────
    df = pd.read_csv(LANG_TO_DATA_PATH[lang])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{LANG_TO_DATA_PATH[lang]} is missing columns: {missing}")
    df = df[REQUIRED_COLS].copy().reset_index(drop=True)
    num_rows = len(df)

    # ── Resume from checkpoint ────────────────────────────────────────────
    completed_indices: set[int] = set()
    if os.path.exists(ckpt_path):
        ckpt_existing = pd.read_csv(ckpt_path)
        completed_indices = set(range(len(ckpt_existing)))
        print(f"[{lang.upper()}] Resuming from Q{len(ckpt_existing)+1} "
              f"({len(ckpt_existing)} rows already done)")

    remaining_indices = [i for i in range(num_rows) if i not in completed_indices]

    print(f"\n{'='*60}")
    print(f"Running MGSM {N_RUNS}-run majority vote [{MODEL_NAME}] : {full_name} ({lang.upper()})")
    print(f"  {len(remaining_indices)} questions remaining  |  {num_rows} total")
    print(f"{'='*60}")

    if remaining_indices:
        # ── Build chat-formatted prompts ──────────────────────────────────
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user",
                  "content": LANG_TO_INSTRUCTIONS[lang].format(
                      input=df.iloc[i]["question"]
                  )}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in remaining_indices
        ]

        # ── N_RUNS batched generations ────────────────────────────────────
        # run_outputs[run] = list of texts aligned with remaining_indices
        run_outputs: list[list[str]] = []
        for run in range(N_RUNS):
            print(f"  Run {run+1}/{N_RUNS} — generating {len(chat_prompts)} responses …")
            vllm_out = llm.generate(chat_prompts, sampling_params)
            run_outputs.append([o.outputs[0].text.strip() for o in vllm_out])

        # ── Write checkpoint rows ─────────────────────────────────────────
        write_header = not os.path.exists(ckpt_path)
        with open(ckpt_path, "a", newline="", encoding="utf-8") as ckpt_file:
            writer = csv.DictWriter(
                ckpt_file, fieldnames=CKPT_COLS, quoting=csv.QUOTE_ALL
            )
            if write_header:
                writer.writeheader()

            for pos, idx in enumerate(remaining_indices):
                row      = df.iloc[idx]
                expected = str(row["answer"]).strip()
                q_label  = f"[{lang.upper()}] Q{idx+1}"

                run_results = []
                for run in range(N_RUNS):
                    cot               = run_outputs[run][pos]
                    extracted, method = parse_answer(cot, answer_prefix, lang)
                    correct           = answers_are_equal(extracted, expected)
                    run_results.append({
                        "cot": cot, "extracted": extracted,
                        "parse_method": method, "correct": correct,
                    })
                    print(
                        f"  {q_label} Run {run+1}: "
                        f"extracted={extracted!r:>10} | expected={expected} | "
                        f"method={method} | {'OK' if correct else 'WRONG'}"
                    )

                answers_this_q = [r["extracted"] for r in run_results]
                vote, status   = majority_vote(answers_this_q)
                flag = " [ALL DIFFER — falling back to run 1]" \
                       if status == "all_differ" else ""
                print(f"  → [{status}] vote={vote!r} | expected={expected}{flag}")

                ckpt_row = {
                    "question": row["question"],
                    "answer":   row["answer"],
                    **{f"cot_run{r+1}":          run_results[r]["cot"]          for r in range(N_RUNS)},
                    **{f"extracted_run{r+1}":    run_results[r]["extracted"]    for r in range(N_RUNS)},
                    **{f"parse_method_run{r+1}": run_results[r]["parse_method"] for r in range(N_RUNS)},
                    **{f"correct_run{r+1}":      run_results[r]["correct"]      for r in range(N_RUNS)},
                    "majority_vote":    vote,
                    "vote_status":      status,
                    "majority_correct": answers_are_equal(vote, expected),
                }
                writer.writerow(ckpt_row)
                ckpt_file.flush()

    # ── Build final dataframe from checkpoint ─────────────────────────────
    result_df = pd.read_csv(ckpt_path)

    # Parse method breakdown
    all_methods: list[str] = []
    for r in range(N_RUNS):
        all_methods.extend(result_df[f"parse_method_run{r+1}"].tolist())
    method_counts = Counter(all_methods)
    total_calls   = len(result_df) * N_RUNS
    failed_calls  = method_counts.get("failed", 0)

    print(f"\n{'─'*50}")
    print(f"[{lang.upper()}] PARSE METHOD BREAKDOWN ({total_calls} total calls):")
    for method, count in method_counts.most_common():
        print(f"  {method:20s}: {count:3d}  ({count/total_calls*100:.1f}%)")
    print(f"  → Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    # Per-run + majority accuracy
    total = len(result_df)
    print(f"\n[{lang.upper()}] ACCURACY:")
    for r in range(N_RUNS):
        acc = result_df[f"correct_run{r+1}"].sum()
        print(f"  Run {r+1}: {acc}/{total} = {acc/total:.1%}")
    majority_acc = int(result_df["majority_correct"].sum())
    print(f"  Majority vote: {majority_acc}/{total} = {majority_acc/total:.1%}")

    # Vote status breakdown
    status_counts = result_df["vote_status"].value_counts()
    print(f"\n[{lang.upper()}] VOTE STATUS:")
    for status, count in status_counts.items():
        print(f"  {status:12s}: {count} questions")

    # ── Failed parse details ──────────────────────────────────────────────
    failed_rows: list[dict] = []
    for i, row in result_df.iterrows():
        for r in range(N_RUNS):
            if row[f"parse_method_run{r+1}"] == "failed":
                failed_rows.append({
                    "lang": lang,
                    "q_idx": i + 1,
                    "run": r + 1,
                    "question": str(row["question"])[:100],
                    "expected": row["answer"],
                    "cot_tail": "\n".join(
                        str(row[f"cot_run{r+1}"]).strip().split("\n")[-6:]
                    ),
                })

    if failed_rows:
        print(f"\n[{lang.upper()}] FAILED PARSE DETAILS:")
        for fr in failed_rows:
            print(f"  Q{fr['q_idx']} Run{fr['run']} | expected={fr['expected']}")
            print(f"  Question: {fr['question']}")
            print(f"  --- Last 6 lines of CoT ---")
            print(fr["cot_tail"])
            print(f"  {'─'*40}")
    else:
        print(f"\n[{lang.upper()}] All calls parsed successfully.")

    # ── Save final CSV ────────────────────────────────────────────────────
    result_df = result_df[CKPT_COLS]   # enforce column order
    result_df.to_csv(output_path, index=False, quoting=1)
    print(f"\nSaved: {output_path}")

    # Remove checkpoint now that final CSV is written
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return {
        "language":       full_name,
        "code":           lang,
        "correct":        majority_acc,
        "total":          total,
        "accuracy":       f"{majority_acc/total:.1%}",
        "parse_failures": failed_calls,
        "output_file":    output_path,
    }


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    summary_rows: list[dict] = []

    for lang in ALL_LANGUAGES:
        result = run_inference_for_lang(lang)
        summary_rows.append(result)

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_rows).sort_values("language")
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 60)
    print(f"SUMMARY (majority vote) — {MODEL_NAME}")
    print("=" * 60)
    print(summary_df.to_string(index=False))

    total_failures = summary_df["parse_failures"].sum()
    print(f"\nTotal parse failures across all languages: {total_failures}")
    print(f"\nSummary saved: {summary_path}")
    print(f"All outputs in: {OUTPUT_DIR}")
    if total_failures > 0:
        print("Inspect CSVs with parse_method == 'failed' to improve parsing.")
