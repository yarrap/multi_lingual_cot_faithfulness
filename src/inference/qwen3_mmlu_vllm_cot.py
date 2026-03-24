"""
qwen3_mmlu_vllm_cot.py
======================
MMLU evaluation for Qwen3 using vLLM + majority voting.
Output format: CSV (mirrors Doc 2 / Gemma MMLU structure).

Expected input files (one per language):
    mmlu_en.csv, mmlu_bn.csv, mmlu_sw.csv, mmlu_te.csv, mmlu_zh.csv
Each must contain columns: question, A, B, C, D, answer

Output per language  : <OUTPUT_DIR>/cot_<lang>.csv
Summary              : <OUTPUT_DIR>/summary.csv
"""

# ─────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────
import os
import re
import csv
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────
# CONFIG  ← edit these to match your setup
# ─────────────────────────────────────────────
MODEL_NAME  = "Qwen/Qwen3-4B"       # swap to any HF Qwen3 variant
OUTPUT_DIR  = "qwen3_mmlu_outputs"
INPUT_FILES = [
    "mmlu_en.csv",
    "mmlu_bn.csv",
    "mmlu_sw.csv",
    "mmlu_te.csv",
    "mmlu_zh.csv",
]

N_RUNS       = 3      # majority vote across this many independent generations
MAX_TOKENS   = 512
TEMPERATURE  = 0.7    # >0 gives diversity across runs; set 0.0 for greedy
TOP_P        = 0.95
GPU_MEM_UTIL = 0.90

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PROMPT TEMPLATES
# ─────────────────────────────────────────────
LANG_TO_INSTRUCTIONS: dict[str, str] = {
    "en": (
        "The following are multiple choice questions (with answers) about general "
        "knowledge. Think step by step and then finish your answer with "
        "\"the answer is (X)\" where X is the correct letter choice.\n\n"
        "Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    ),
    "bn": (
        "নিচে সাধারণ জ্ঞান সম্পর্কে বহুনির্বাচনী প্রশ্ন (উত্তর সহ) রয়েছে। "
        "ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার উত্তরটি "
        "\"উত্তর হল (X)\" দিয়ে শেষ করুন যেখানে X হল সঠিক অক্ষর পছন্দ।\n\n"
        "প্রশ্ন: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    ),
    "sw": (
        "Zifuatazo ni maswali ya chaguo nyingi (pamoja na majibu) kuhusu maarifa "
        "ya jumla. Fikiria hatua kwa hatua na kisha maliza jibu lako na "
        "\"jibu ni (X)\" ambapo X ni chaguo sahihi la herufi.\n\n"
        "Swali: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    ),
    "te": (
        "కింది వాటిలో సాధారణ జ్ఞానం గురించి బహుళ ఎంపిక ప్రశ్నలు (సమాధానాలతో) "
        "ఉన్నాయి. దశలవారీగా ఆలోచించండి మరియు తరువాత మీ సమాధానాన్ని "
        "\"సమాధానం (X)\" తో ముగించండి, ఇక్కడ X సరైన అక్షర ఎంపిక.\n\n"
        "ప్రశ్న: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    ),
    "zh": (
        "以下是关于常识的多项选择题（带答案）。逐步思考，然后用"
        "\"答案是 (X)\"结束你的回答，其中 X 是正确的字母选择。\n\n"
        "问题: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}"
    ),
}

LANG_TO_ANSWER_PREFIX: dict[str, str] = {
    "en": "the answer is",
    "bn": "উত্তর হল",
    "sw": "jibu ni",
    "te": "సమాధానం",
    "zh": "答案是",
}

_FULL_NAME_TO_CODE: dict[str, str] = {
    "english": "en",
    "bengali": "bn",
    "swahili": "sw",
    "telugu":  "te",
    "chinese": "zh",
}

REQUIRED_COLS = ["question", "A", "B", "C", "D", "answer"]

# Build interleaved iteration columns (matches Doc 2 column order)
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

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def infer_lang(filename: str) -> str:
    """Infer 2-letter language code from filename."""
    base = os.path.basename(filename).lower()
    stem = os.path.splitext(base)[0]
    for full, code in _FULL_NAME_TO_CODE.items():
        if full in stem:
            return code
    m = re.search(r"(?:^|[_\-])(en|bn|sw|te|zh)(?:[_\-]|$)", stem)
    if m:
        return m.group(1)
    raise ValueError(
        f"Cannot infer language from: {filename}\n"
        "Expected one of: mmlu_en.csv / mmlu_english.csv / mmlu_swahili.csv …"
    )


def parse_answer(cot_text: str, answer_prefix: str, lang: str = "") -> tuple[str, str]:
    """
    Extract A/B/C/D letter from chain-of-thought output.
    Returns (answer_letter, parse_method).
    Includes the zh-specific pre-pass ported from the MGSM Qwen3 script.
    """
    if not isinstance(cot_text, str):
        return "", "failed"

    lines = cot_text.strip().split("\n")

    # ── Chinese-specific pre-pass ──────────────────────────────────────────
    if lang == "zh":
        def norm(s: str) -> str:
            return s.replace("：", ":").replace("︓", ":")

        norm_prefix = norm(answer_prefix)

        # 1. Exact prefix match (scan from bottom)
        for line in reversed(lines):
            clean = line.replace("**", "").strip()
            if answer_prefix in clean:
                tail = clean[clean.find(answer_prefix):]
                m = re.search(r"\(?([A-Da-d])\)?", tail)
                if m:
                    return m.group(1).upper(), "zh_primary"

        # 2. Normalised colon prefix
        for line in reversed(lines):
            clean = norm(line.replace("**", "").strip())
            if norm_prefix in clean:
                tail = clean[clean.find(norm_prefix):]
                m = re.search(r"\(?([A-Da-d])\)?", tail)
                if m:
                    return m.group(1).upper(), "zh_fuzzy"

        # 3. Bold answer pattern  **答案** / **答案是**
        for line in reversed(lines):
            if re.search(r'\*\*.*答案.*\*\*', line):
                m = re.search(r"\(?([A-Da-d])\)?", line)
                if m:
                    return m.group(1).upper(), "zh_bold_prefix"

    # ── Primary: "answer_prefix (X)" or "answer_prefix X" ─────────────────
    pattern = rf"{re.escape(answer_prefix)}\s*\(?([A-Da-d])\)?"
    match = re.search(pattern, cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "primary"

    # ── Answer on the next line after the prefix ───────────────────────────
    for i, line in enumerate(lines):
        if answer_prefix.lower() in line.lower() and i + 1 < len(lines):
            m = re.search(r"\(?([A-Da-d])\)?", lines[i + 1])
            if m:
                return m.group(1).upper(), "next_line"

    # ── Last line is a bare letter  A / (B) / [C] ─────────────────────────
    for line in reversed(lines):
        m = re.match(r"^[(\[]?([A-Da-d])[)\].]?\s*$", line.strip(), re.IGNORECASE)
        if m:
            return m.group(1).upper(), "last_line"

    # ── Bold letter  **A** / **(B)** ──────────────────────────────────────
    for line in reversed(lines[-5:]):
        bold_m = re.findall(r'\*\*\(?([A-Da-d])\)?\*\*', line, re.IGNORECASE)
        if bold_m:
            return bold_m[-1].upper(), "fallback_bold"

    # ── Last standalone letter anywhere ───────────────────────────────────
    matches = re.findall(r"\b([A-D])\b", cot_text.upper())
    if matches:
        return matches[-1], "last_letter"

    return "", "failed"


def answers_equal(extracted: str, expected: str) -> bool:
    return str(extracted).strip().upper() == str(expected).strip().upper()


def majority_vote(answers: list[str]) -> tuple[str, str]:
    """
    Return (most_common_answer, vote_status).
    status: 'unanimous' | 'majority' | 'all_differ' | 'no_answer'
    Falls back to run-1 answer when all differ (mirrors MGSM behaviour).
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


def build_prompt(row: pd.Series, lang: str) -> str:
    return LANG_TO_INSTRUCTIONS[lang].format(
        question=row["question"],
        A=row["A"], B=row["B"], C=row["C"], D=row["D"],
    )

# ─────────────────────────────────────────────
# LOAD MODEL  (once, shared across all languages)
# ─────────────────────────────────────────────
print(f"\nLoading tokenizer : {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading vLLM engine: {MODEL_NAME}")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=GPU_MEM_UTIL,
)

sampling_params = SamplingParams(
    temperature=TEMPERATURE,
    top_p=TOP_P,
    max_tokens=MAX_TOKENS,
)
print("Model loaded.\n")

# ─────────────────────────────────────────────
# PER-LANGUAGE INFERENCE
# ─────────────────────────────────────────────

def run_inference_for_lang(input_file: str) -> dict:
    lang          = infer_lang(input_file)
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]
    output_path   = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
    ckpt_path     = os.path.join(OUTPUT_DIR, f"cot_{lang}_checkpoint.csv")

    # ── Skip if already complete ──────────────────────────────────────────
    if os.path.exists(output_path):
        existing = pd.read_csv(output_path)
        if len(existing) >= 100:   # adjust if your MMLU slice is larger
            print(f"[{lang.upper()}] Already complete ({len(existing)} rows) — skipping.")
            majority_acc = int(existing["majority_correct"].sum())
            total        = len(existing)
            return {
                "language":       lang,
                "correct":        majority_acc,
                "total":          total,
                "accuracy":       f"{majority_acc/total:.1%}",
                "parse_failures": 0,
                "output_file":    output_path,
            }

    # ── Load & validate ───────────────────────────────────────────────────
    df = pd.read_csv(input_file)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{input_file} is missing columns: {missing}")
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
    print(f"MMLU {N_RUNS}-run majority vote [{MODEL_NAME}] : {lang.upper()}")
    print(f"  {len(remaining_indices)} questions remaining  |  {num_rows} total")
    print(f"{'='*60}")

    if remaining_indices:
        # ── Build chat-formatted prompts for remaining rows ───────────────
        chat_prompts = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": build_prompt(df.iloc[i], lang)}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for i in remaining_indices
        ]

        # ── N_RUNS batched generations ────────────────────────────────────
        # run_outputs[run] = list of texts, aligned with remaining_indices
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
                expected = str(row["answer"]).strip().upper()
                q_label  = f"[{lang.upper()}] Q{idx+1}"

                run_results = []
                for run in range(N_RUNS):
                    cot               = run_outputs[run][pos]
                    extracted, method = parse_answer(cot, answer_prefix, lang)
                    correct           = answers_equal(extracted, expected)
                    run_results.append({
                        "cot": cot, "extracted": extracted,
                        "parse_method": method, "correct": correct,
                    })
                    print(
                        f"  {q_label} Run {run+1}: "
                        f"extracted={extracted!r:>2} | expected={expected} | "
                        f"method={method} | {'OK' if correct else 'WRONG'}"
                    )

                answers_this_q = [r["extracted"] for r in run_results]
                vote, status   = majority_vote(answers_this_q)
                flag = " (ALL DIFFER — falling back to run 1)" \
                       if status == "all_differ" else ""
                print(f"  -> [{status}] vote={vote!r} | expected={expected}{flag}")

                ckpt_row = {
                    "question": row["question"],
                    "A": row["A"], "B": row["B"],
                    "C": row["C"], "D": row["D"],
                    "answer": row["answer"],
                    **{f"cot_run{r+1}":          run_results[r]["cot"]          for r in range(N_RUNS)},
                    **{f"extracted_run{r+1}":    run_results[r]["extracted"]    for r in range(N_RUNS)},
                    **{f"parse_method_run{r+1}": run_results[r]["parse_method"] for r in range(N_RUNS)},
                    **{f"correct_run{r+1}":      run_results[r]["correct"]      for r in range(N_RUNS)},
                    "majority_vote":    vote,
                    "vote_status":      status,
                    "majority_correct": answers_equal(vote, expected),
                }
                writer.writerow(ckpt_row)
                ckpt_file.flush()

    # ── Build final dataframe from checkpoint ─────────────────────────────
    result_df = pd.read_csv(ckpt_path)

    # Parse method breakdown (mirrors MGSM logging exactly)
    all_methods = []
    for r in range(N_RUNS):
        all_methods.extend(result_df[f"parse_method_run{r+1}"].tolist())
    method_counts = Counter(all_methods)
    total_calls   = len(result_df) * N_RUNS
    failed_calls  = method_counts.get("failed", 0)

    print(f"\n{'─'*50}")
    print(f"[{lang.upper()}] PARSE METHOD BREAKDOWN ({total_calls} total calls):")
    for method, count in method_counts.most_common():
        print(f"  {method:20s}: {count:3d}  ({count/total_calls*100:.1f}%)")
    print(f"  -> Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    # Per-run + majority accuracy
    total = len(result_df)
    print(f"\n[{lang.upper()}] ACCURACY:")
    for r in range(N_RUNS):
        acc = result_df[f"correct_run{r+1}"].sum()
        print(f"  Run {r+1}: {acc}/{total} = {acc/total:.1%}")
    majority_acc = int(result_df["majority_correct"].sum())
    print(f"  Majority vote: {majority_acc}/{total} = {majority_acc/total:.1%}")

    # Vote status breakdown (mirrors Doc 2 logging)
    status_counts = result_df["vote_status"].value_counts()
    print(f"\n[{lang.upper()}] VOTE STATUS:")
    for status, count in status_counts.items():
        print(f"  {status:12s}: {count} questions")

    # ── Save final CSV (column order matches Doc 2) ───────────────────────
    result_df = result_df[CKPT_COLS]   # enforce column order
    result_df.to_csv(output_path, index=False, quoting=1)
    print(f"\nSaved: {output_path}")

    # Remove checkpoint now that final CSV is written
    os.remove(ckpt_path)

    return {
        "language":       lang,
        "correct":        majority_acc,
        "total":          total,
        "accuracy":       f"{majority_acc/total:.1%}",
        "parse_failures": failed_calls,
        "output_file":    output_path,
    }

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    summary_rows = []

    for input_file in INPUT_FILES:
        print("\n" + "=" * 80)
        print("Processing:", input_file)
        result = run_inference_for_lang(input_file)
        summary_rows.append(result)

    # ── Summary CSV ───────────────────────────────────────────────────────
    summary_df   = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 80)
    print(f"DONE — {MODEL_NAME}")
    print(summary_df.to_string(index=False))
    print(f"\nSummary saved : {summary_path}")
    print(f"All outputs in: {OUTPUT_DIR}")
