
# ============================================================
# 3) GEMMA 3 4B + vLLM GENERATION WITH MAJORITY VOTING
#    Output files will be:
#    cot_en.csv, cot_bn.csv, cot_sw.csv, cot_te.csv, cot_zh.csv
# ============================================================

import os
import re
import pandas as pd
from collections import Counter

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "gemma_mmlu_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

USE_SAMPLING = True
TEMPERATURE = 0.0     # Non-zero for diverse iterations
TOP_P = 0.95

MAX_TOKENS = 500
NUM_ITERATIONS = 3

# ---------------------------
# PROMPT TEMPLATES
# ---------------------------
LANG_TO_INSTRUCTIONS = {
    "en": """The following are multiple choice questions (with answers) about general knowledge. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "bn": """নিচে সাধারণ জ্ঞান সম্পর্কে বহুনির্বাচনী প্রশ্ন (উত্তর সহ) রয়েছে। ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার উত্তরটি "উত্তর হল (X)" দিয়ে শেষ করুন যেখানে X হল সঠিক অক্ষর পছন্দ।

প্রশ্ন: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "sw": """Zifuatazo ni maswali ya chaguo nyingi (pamoja na majibu) kuhusu maarifa ya jumla. Fikiria hatua kwa hatua na kisha maliza jibu lako na "jibu ni (X)" ambapo X ni chaguo sahihi la herufi.

Swali: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "te": """కింది వాటిలో సాధారణ జ్ఞానం గురించి బహుళ ఎంపిక ప్రశ్నలు (సమాధానాలతో) ఉన్నాయి. దశలవారీగా ఆలోచించండి మరియు తరువాత మీ సమాధానాన్ని "సమాధానం (X)" తో ముగించండి, ఇక్కడ X సరైన అక్షర ఎంపిక.

ప్రశ్న: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "zh": """以下是关于常识的多项选择题（带答案）。逐步思考，然后用"答案是 (X)"结束你的回答，其中 X 是正确的字母选择。

问题: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": "the answer is",
    "bn": "উত্তর হল",
    "sw": "jibu ni",
    "te": "సమాధానం",
    "zh": "答案是",
}

REQUIRED_COLS = ["question", "A", "B", "C", "D", "answer"]
LANG_CODES = ["en", "bn", "sw", "te", "zh"]

# ---------------------------
# HELPERS
# ---------------------------
FULL_NAME_TO_CODE = {
    "english": "en",
    "bengali": "bn",
    "swahili": "sw",
    "telugu":  "te",
    "chinese": "zh",
}

def infer_lang_from_filename(filename: str) -> str:
    """
    Handles filenames like:
      mmlu_en.csv, mmlu_bn.csv
      mmlu_english.csv, mmlu_swahili.csv, mmlu_telugu.csv, etc.
    """
    base = os.path.basename(filename).lower()
    stem = os.path.splitext(base)[0]

    for full_name, code in FULL_NAME_TO_CODE.items():
        if full_name in stem:
            return code

    m = re.search(r"(?:^|[_\-])(en|bn|sw|te|zh)(?:[_\-]|$)", stem)
    if m:
        return m.group(1)

    raise ValueError(
        f"Could not infer language from filename: {filename}\n"
        f"Expected formats: mmlu_en.csv, mmlu_english.csv, mmlu_swahili.csv, etc."
    )

def parse_answer(cot_text: str, answer_prefix: str) -> tuple[str, str]:
    """
    Extract A/B/C/D from model output.
    Returns (answer, parse_method) to mirror MGSM code.
    """
    if not isinstance(cot_text, str):
        return "", "failed"

    # Primary: "the answer is (X)" / "উত্তর হল (X)" / etc.
    pattern = rf"{re.escape(answer_prefix)}\s*\(?([A-Da-d])\)?"
    match = re.search(pattern, cot_text, re.IGNORECASE)
    if match:
        return match.group(1).upper(), "primary"

    # Fallback: last line is just A/B/C/D
    lines = cot_text.strip().split("\n")
    for line in reversed(lines):
        line = line.strip()
        m = re.match(r"^[(\[]?([A-Da-d])[)\].]?\s*$", line, re.IGNORECASE)
        if m:
            return m.group(1).upper(), "last_line"

    # Last fallback: last standalone option letter anywhere
    matches = re.findall(r"\b([A-D])\b", cot_text.upper())
    if matches:
        return matches[-1], "last_letter"

    return "", "failed"

def majority_vote(answers: list[str]) -> tuple[str, str]:
    """
    Return (most_common_answer, vote_status).
    vote_status is one of: 'unanimous', 'majority', 'all_differ'.
    Empty strings are excluded; if all empty, returns ("", "no_answer").
    Falls back to first answer when all differ (mirrors MGSM behaviour).
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
        return valid[0], "all_differ"   # fall back to run-1 answer

def validate_input_df(df: pd.DataFrame, filename: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

def build_raw_prompt(row, lang: str) -> str:
    template = LANG_TO_INSTRUCTIONS[lang]
    return template.format(
        question=row["question"],
        A=row["A"],
        B=row["B"],
        C=row["C"],
        D=row["D"],
    )

# ---------------------------
# LOAD TOKENIZER + VLLM
# ---------------------------
print(f"Loading tokenizer for {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading vLLM model for {MODEL_NAME} ...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    gpu_memory_utilization=0.90,
)

if USE_SAMPLING:
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
    )
else:
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=MAX_TOKENS,
    )

print("Model loaded successfully.")

# ---------------------------
# MAIN LOOP
# ---------------------------
summary_rows = []

uploaded_files = [
    "/content/mmlu_en.csv",
    "/content/mmlu_bn.csv",
    "/content/mmlu_sw.csv",
    "/content/mmlu_te.csv",
    "/content/mmlu_zh.csv",
]

for input_file in uploaded_files:
    print("\n" + "=" * 80)
    print("Processing:", input_file)

    lang = infer_lang_from_filename(input_file)
    answer_prefix = LANG_TO_ANSWER_PREFIX[lang]

    df = pd.read_csv(input_file)
    validate_input_df(df, input_file)
    df = df[REQUIRED_COLS].copy()

    # Build prompts once — reused across all iterations
    raw_prompts = [build_raw_prompt(row, lang) for _, row in df.iterrows()]
    prompts = []
    for prompt in raw_prompts:
        chat_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(chat_prompt)

    # ---------------------------
    # 3 ITERATIONS
    # ---------------------------
    # run_data[run] = {"cots": [...], "extracted": [...], "parse_methods": [...], "corrects": [...]}
    run_data = {
        run: {"cots": [], "extracted": [], "parse_methods": [], "corrects": []}
        for run in range(NUM_ITERATIONS)
    }

    for iteration in range(NUM_ITERATIONS):
        print(f"  Iteration {iteration+1}/{NUM_ITERATIONS} — generating {len(prompts)} responses ...")
        outputs = llm.generate(prompts, sampling_params)

        for out in outputs:
            text = out.outputs[0].text.strip()
            extracted, method = parse_answer(text, answer_prefix)
            run_data[iteration]["cots"].append(text)
            run_data[iteration]["extracted"].append(extracted)
            run_data[iteration]["parse_methods"].append(method)
            run_data[iteration]["corrects"].append(
                extracted == df["answer"].iloc[len(run_data[iteration]["corrects"])].strip()
            )

    # ---------------------------
    # MAJORITY VOTING + STATS
    # ---------------------------
    num_rows = len(df)

    for i in range(NUM_ITERATIONS):
        df[f"cot_run{i+1}"]          = run_data[i]["cots"]
        df[f"extracted_run{i+1}"]    = run_data[i]["extracted"]
        df[f"parse_method_run{i+1}"] = run_data[i]["parse_methods"]
        df[f"correct_run{i+1}"]      = run_data[i]["corrects"]

    vote_results = [
        majority_vote([run_data[run]["extracted"][row_idx] for run in range(NUM_ITERATIONS)])
        for row_idx in range(num_rows)
    ]
    df["majority_vote"]    = [v for v, _ in vote_results]
    df["vote_status"]      = [s for _, s in vote_results]
    df["majority_correct"] = (
        df["majority_vote"].astype(str).str.strip()
        == df["answer"].astype(str).str.strip()
    )

    # ---------------------------
    # PARSE METHOD BREAKDOWN (mirrors MGSM logging)
    # ---------------------------
    all_methods = []
    for run in range(NUM_ITERATIONS):
        all_methods.extend(run_data[run]["parse_methods"])
    method_counts = Counter(all_methods)
    total_calls = num_rows * NUM_ITERATIONS
    failed_calls = method_counts.get("failed", 0)

    print(f"\n[{lang.upper()}] PARSE METHOD BREAKDOWN ({total_calls} total calls):")
    for method, count in method_counts.most_common():
        print(f"  {method:16s}: {count:3d}  ({count/total_calls*100:.1f}%)")
    print(f"  -> Parse failures: {failed_calls}/{total_calls} ({failed_calls/total_calls*100:.1f}%)")

    # Per-run and majority accuracy
    print(f"\n[{lang.upper()}] ACCURACY:")
    for run in range(NUM_ITERATIONS):
        acc = sum(run_data[run]["corrects"])
        print(f"  Run {run+1}: {acc}/{num_rows} = {acc/num_rows:.1%}")
    majority_acc = int(df["majority_correct"].sum())
    print(f"  Majority vote: {majority_acc}/{num_rows} = {majority_acc/num_rows:.1%}")

    # Vote status breakdown
    status_counts = df["vote_status"].value_counts()
    for status, count in status_counts.items():
        print(f"  {status:12s}: {count} questions")

    # ---------------------------
    # SAVE
    # ---------------------------
    iter_cols = []
    for i in range(NUM_ITERATIONS):
        iter_cols += [
            f"cot_run{i+1}",
            f"extracted_run{i+1}",
            f"parse_method_run{i+1}",
            f"correct_run{i+1}",
        ]

    df = df[
        ["question", "A", "B", "C", "D", "answer"]
        + iter_cols
        + ["majority_vote", "vote_status", "majority_correct"]
    ]

    output_file = os.path.join(OUTPUT_DIR, f"cot_{lang}.csv")
    df.to_csv(output_file, index=False, quoting=1)
    print(f"\nSaved: {output_file}")

    summary_rows.append({
        "language": lang,
        "correct": majority_acc,
        "total": num_rows,
        "accuracy": f"{majority_acc/num_rows:.1%}",
        "parse_failures": failed_calls,
        "output_file": output_file,
    })

# ---------------------------
# SAVE SUMMARY
# ---------------------------
summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("DONE")
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: {summary_path}")
print(f"All outputs are in: {OUTPUT_DIR}")
