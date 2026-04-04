"""
gemma3_mgsm_vllm_cot.py
========================
MGSM CoT evaluation for Gemma-3-4B-IT using vLLM + 3-run majority voting.
"""

import os
import re
import csv
import pandas as pd
from collections import Counter
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────
HF_MODEL_ID  = "google/gemma-3-4b-it"
MODEL_NAME   = "gemma3-4b-vllm"
OUTPUT_DIR   = f"/content/results/{MODEL_NAME}"

ALL_LANGUAGES = ["en", "sw", "te", "zh", "bn"]

N_RUNS       = 3
MAX_TOKENS   = 4096      # changed: was 512
TEMPERATURE  = 0.6       # changed: was 0.0
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
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",

    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}""",

    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",

    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}""",

    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer:",
    "bn": "উত্তর:",
    "sw": "Jibu:",
    "te": "సమాధానం:",
    "zh": "答案:",
}

REQUIRED_COLS = ["question", "answer"]

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
# IMPROVED ANSWER PARSER
# ─────────────────────────────────────────────────────────────────

def normalize_number(s: str) -> str:
    """
    Normalize a numeric string:
      - Remove thousands separators (, or ' or space-as-thousands)
      - Strip trailing dot
      - Convert unicode digits (Bengali, Telugu, Chinese, etc.) to ASCII
    """
    if not isinstance(s, str):
        return ""

    # Unicode digit normalization — covers Bengali, Telugu, Arabic-Indic, etc.
    result = []
    for ch in s:
        cp = ord(ch)
        # Bengali digits: 0x09E6–0x09EF
        if 0x09E6 <= cp <= 0x09EF:
            result.append(str(cp - 0x09E6))
        # Telugu digits: 0x0C66–0x0C6F
        elif 0x0C66 <= cp <= 0x0C6F:
            result.append(str(cp - 0x0C66))
        # Arabic-Indic digits: 0x0660–0x0669
        elif 0x0660 <= cp <= 0x0669:
            result.append(str(cp - 0x0660))
        # Extended Arabic-Indic: 0x06F0–0x06F9
        elif 0x06F0 <= cp <= 0x06F9:
            result.append(str(cp - 0x06F0))
        # Chinese/fullwidth digits: 0xFF10–0xFF19
        elif 0xFF10 <= cp <= 0xFF19:
            result.append(str(cp - 0xFF10))
        else:
            result.append(ch)
    s = "".join(result)

    # Remove thousands separators: comma, apostrophe, and unicode thin-space
    s = re.sub(r"[,'\u202f\u00a0]", "", s)
    # Strip currency / percent symbols that may wrap the number
    s = s.replace("$", "").replace("₹", "").replace("€", "").replace("%", "").strip()
    # Remove trailing dot (e.g. "42.")
    s = s.rstrip(".")
    return s


def extract_numbers(text: str) -> list[str]:
    """Return all numeric tokens (including decimals) from text, normalized."""
    cleaned = normalize_number(text)
    return re.findall(r"-?\d+\.?\d*", cleaned)


def extract_boxed(line: str) -> str:
    """Extract value from \\boxed{...}, resolving '=' chains."""
    m = re.search(r'\\boxed\{([^}]+)\}', line)
    if not m:
        return ""
    inner = m.group(1)
    if "=" in inner:
        inner = inner.split("=")[-1]
    nums = extract_numbers(inner)
    return nums[-1] if nums else ""


def extract_inline_latex(line: str) -> str:
    """Extract number from \\( ... \\) or \\[ ... \\] inline LaTeX."""
    m = re.search(r'\\\(([^)]+)\\\)|\\\[([^\]]+)\\\]', line)
    if not m:
        return ""
    inner = (m.group(1) or m.group(2) or "").strip()
    nums = extract_numbers(inner)
    return nums[-1] if nums else ""


def find_prefix_line(lines: list[str], prefix: str) -> int:
    """
    Return index of the LAST line containing prefix (stripped of markdown bold).
    Returns -1 if not found.
    """
    for i in range(len(lines) - 1, -1, -1):
        clean = lines[i].replace("**", "").replace("__", "").strip()
        if prefix in clean:
            return i
    return -1


def parse_answer(cot_text: str, answer_prefix: str, lang: str = "") -> tuple[str, str]:
    """
    Multi-stage answer extractor. Returns (answer_str, parse_method).

    Stage order (all languages):
      1. primary        — prefix + number on same line (last such line)
      2. next_line_box  — prefix alone, \\boxed{} on next line
      3. next_line_num  — prefix alone, plain number on next line
      4. prefix_inline  — prefix line contains inline LaTeX \\(...\\)
      5. bold_answer    — **<number>** pattern near end
      6. boxed_tail     — \\boxed{} in last 8 lines
      7. inline_tail    — inline LaTeX in last 8 lines
      8. dollar_math    — $number$ pattern in last 8 lines
      9. therefore      — "therefore"/"thus"/"hence" + number in last 8 lines
     10. last_sentence  — last sentence of CoT containing a number
     11. last_number    — absolute last number in entire CoT
     12. failed
    """
    if not isinstance(cot_text, str) or not cot_text.strip():
        return "", "failed"

    lines = cot_text.strip().split("\n")
    tail  = lines[-8:]  # last 8 lines for fallback stages

    # ── Chinese-specific normalization of full-width colons ───────────────
    def normalize_zh(s: str) -> str:
        return s.replace("：", ":").replace("︓", ":").replace("﹕", ":")

    effective_prefix = normalize_zh(answer_prefix) if lang == "zh" else answer_prefix

    def line_has_prefix(line: str) -> bool:
        clean = line.replace("**", "").replace("__", "").strip()
        if lang == "zh":
            return effective_prefix in normalize_zh(clean)
        return effective_prefix in clean

    # ── Stage 1: primary ─────────────────────────────────────────────────
    for line in reversed(lines):
        if line_has_prefix(line):
            nums = extract_numbers(line)
            if nums:
                return nums[-1], "primary"

    # ── Stage 2: next_line_box ───────────────────────────────────────────
    for i, line in enumerate(lines):
        if line_has_prefix(line) and i + 1 < len(lines):
            val = extract_boxed(lines[i + 1])
            if val:
                return val, "next_line_box"

    # ── Stage 3: next_line_num ───────────────────────────────────────────
    for i, line in enumerate(lines):
        if line_has_prefix(line) and i + 1 < len(lines):
            nums = extract_numbers(lines[i + 1])
            if nums:
                return nums[0], "next_line_num"

    # ── Stage 4: prefix_inline (LaTeX on same prefix line) ───────────────
    for line in reversed(lines):
        if line_has_prefix(line):
            val = extract_inline_latex(line)
            if val:
                return val, "prefix_inline"

    # ── Stage 5: bold_answer — **number** anywhere in tail ───────────────
    for line in reversed(tail):
        # Match **digits** with optional surrounding text
        bold_nums = re.findall(r'\*\*\s*(-?\d[\d,\.]*)\s*\*\*', line.replace(",", ""))
        if bold_nums:
            return normalize_number(bold_nums[-1]), "bold_answer"

    # ── Stage 6: boxed_tail ──────────────────────────────────────────────
    for line in reversed(tail):
        val = extract_boxed(line)
        if val:
            return val, "boxed_tail"

    # ── Stage 7: inline_tail ─────────────────────────────────────────────
    for line in reversed(tail):
        val = extract_inline_latex(line)
        if val:
            return val, "inline_tail"

    # ── Stage 8: dollar_math — $number$ in tail ──────────────────────────
    for line in reversed(tail):
        dollar_nums = re.findall(r'\$\s*(-?[\d,\.]+)\s*\$', line)
        if dollar_nums:
            val = normalize_number(dollar_nums[-1])
            if val:
                return val, "dollar_math"

    # ── Stage 9: therefore/thus/hence + number in tail ───────────────────
    conclusion_re = re.compile(
        r'(?:therefore|thus|hence|so|总共|因此|所以|अतः|కాబట్టి|kwa hivyo|nitorina)'
        r'.{0,80}?(-?\d[\d,\.]*)',
        re.IGNORECASE
    )
    for line in reversed(tail):
        m = conclusion_re.search(line)
        if m:
            val = normalize_number(m.group(1))
            if val:
                return val, "therefore"

    # ── Stage 10: last sentence with a number ────────────────────────────
    # Split on sentence boundaries and scan from the end
    sentences = re.split(r'(?<=[.!?।。])\s+', cot_text.strip())
    for sent in reversed(sentences):
        nums = extract_numbers(sent)
        if nums:
            return nums[-1], "last_sentence"

    # ── Stage 11: absolute last number in CoT ────────────────────────────
    all_nums = extract_numbers(cot_text)
    if all_nums:
        return all_nums[-1], "last_number"

    return "", "failed"


# ─────────────────────────────────────────────────────────────────
# VOTING HELPERS
# ─────────────────────────────────────────────────────────────────

def answers_are_equal(extracted: str, expected: str) -> bool:
    """Numeric equality check; falls back to string match."""
    try:
        return (
            float(normalize_number(str(extracted)).strip())
            == float(normalize_number(str(expected)).strip())
        )
    except (ValueError, TypeError):
        return str(extracted).strip() == str(expected).strip()


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
        return valid[0], "all_differ"


# ─────────────────────────────────────────────────────────────────
# LOAD MODEL
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
    output_path   = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
    ckpt_path     = os.path.join(OUTPUT_DIR, f"cot_{lang}_checkpoint.csv")

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

    df = pd.read_csv(LANG_TO_DATA_PATH[lang])
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{LANG_TO_DATA_PATH[lang]} is missing columns: {missing}")
    df = df[REQUIRED_COLS].copy().reset_index(drop=True)
    num_rows = len(df)

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

        run_outputs: list[list[str]] = []
        for run in range(N_RUNS):
            print(f"  Run {run+1}/{N_RUNS} — generating {len(chat_prompts)} responses …")
            vllm_out = llm.generate(chat_prompts, sampling_params)
            run_outputs.append([o.outputs[0].text.strip() for o in vllm_out])

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

    result_df = pd.read_csv(ckpt_path)

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

    total = len(result_df)
    print(f"\n[{lang.upper()}] ACCURACY:")
    for r in range(N_RUNS):
        acc = result_df[f"correct_run{r+1}"].sum()
        print(f"  Run {r+1}: {acc}/{total} = {acc/total:.1%}")
    majority_acc = int(result_df["majority_correct"].sum())
    print(f"  Majority vote: {majority_acc}/{total} = {majority_acc/total:.1%}")

    status_counts = result_df["vote_status"].value_counts()
    print(f"\n[{lang.upper()}] VOTE STATUS:")
    for status, count in status_counts.items():
        print(f"  {status:12s}: {count} questions")

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

    result_df = result_df[CKPT_COLS]
    result_df.to_csv(output_path, index=False, quoting=1)
    print(f"\nSaved: {output_path}")

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
