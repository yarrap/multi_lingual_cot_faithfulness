# ============================================================
# GEMMA 3 4B + vLLM DIRECT MMLU INFERENCE
# Input CSVs must contain:
# question, A, B, C, D, answer
#
# Output CSVs:
# direct_en.csv, direct_bn.csv, direct_sw.csv, direct_te.csv, direct_zh.csv
# ============================================================

import os
import re
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

# ---------------------------
# CONFIG
# ---------------------------
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = "/content/gemma_mmlu_direct_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_TOKENS = 32

REQUIRED_COLS = ["question", "A", "B", "C", "D", "answer"]

LANG_TO_INSTRUCTIONS = {
    "en": """IMPORTANT: Output must be EXACTLY one letter: A, B, C, or D. Do NOT write any explanation. Do NOT write any words. Do NOT solve the problem. Return ONLY one letter.

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer (one letter only):""",

    "bn": """নির্দেশনা অত্যন্ত গুরুত্বপূর্ণ: শুধুমাত্র একটি অক্ষর লিখুন: A, B, C, বা D
কোনো ব্যাখ্যা লিখবেন না
কোনো গণনা দেখাবেন না
কোনো বাক্য লিখবেন না
শুধু একটি অক্ষর লিখুন (যেমন: B)

প্রশ্ন: {question}

A. {A}
B. {B}
C. {C}
D. {D}

উত্তর (শুধু একটি অক্ষর):""",

    "sw": """MAELEKEZO MUHIMU: Andika herufi moja tu: A, B, C, au D
USIANDIKE maelezo yoyote
USIANDIKE sentensi yoyote
Toa herufi moja tu.

Swali: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Jibu (herufi moja tu):""",

    "te": """ముఖ్యమైన సూచనలు: కేవలం ఒక అక్షరం మాత్రమే ఇవ్వండి: A, B, C, లేదా D
ఎటువంటి వివరణ ఇవ్వకండి
ఎటువంటి వాక్యాలు రాయకండి
ఒకే అక్షరం మాత్రమే ఇవ్వండి (ఉదా: B)

ప్రశ్న: {question}

A. {A}
B. {B}
C. {C}
D. {D}

సమాధానం (ఒక అక్షరం మాత్రమే):""",

    "zh": """重要说明：
只能输出一个字母：A、B、C 或 D
不要写任何解释
不要写任何句子
只输出一个字母

问题: {question}

A. {A}
B. {B}
C. {C}
D. {D}

答案（仅一个字母）：""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": ["Answer:"],
    "bn": ["উত্তর:"],
    "sw": ["Jibu:"],
    "te": ["సమాధానం:"],
    "zh": ["答案:", "答案："],
}

LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}

# ---------------------------
# HELPERS
# ---------------------------
def infer_lang_from_filename(filename: str) -> str:
    base = os.path.basename(filename).lower()
    m = re.search(r"(?:^|[_\-])(en|bn|sw|te|zh)(?:[_\-.]|$)", base)
    if m:
        return m.group(1)
    raise ValueError(f"Could not infer language from filename: {filename}")

def validate_input_df(df: pd.DataFrame, filename: str):
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{filename} is missing columns: {missing}")

def build_raw_prompt(row, lang: str) -> str:
    return LANG_TO_INSTRUCTIONS[lang].format(
        question=row["question"],
        A=row["A"],
        B=row["B"],
        C=row["C"],
        D=row["D"],
    )

def parse_answer(response_text: str, lang: str) -> tuple[str, str]:
    if not response_text or str(response_text).strip() == "":
        return "", "failed"

    text = str(response_text).strip()

    # Stage 1: whole response is just A/B/C/D
    if re.fullmatch(r"[A-Da-d]", text):
        return text.upper(), "bare_letter"

    # Stage 2: native answer prefix found
    for prefix in LANG_TO_ANSWER_PREFIX.get(lang, []):
        if prefix in text:
            after = text.split(prefix)[-1].strip()
            m = re.search(r"\b([A-Da-d])\b", after)
            if m:
                return m.group(1).upper(), "prefix_match"

    # Stage 3: fallback — grab last standalone A/B/C/D
    matches = re.findall(r"\b([A-Da-d])\b", text)
    if matches:
        return matches[-1].upper(), "last_letter"

    return "", "failed"

# ---------------------------
# LOAD TOKENIZER + VLLM
# ---------------------------
print(f"Loading tokenizer for {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

print(f"Loading vLLM model for {MODEL_NAME} ...")
llm = LLM(
    model=MODEL_NAME,
    gpu_memory_utilization=0.90,
)

sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=MAX_TOKENS,
)

print("Model loaded successfully.")

# ---------------------------
# MAIN LOOP
# ---------------------------
uploaded_files = [
    "/content/mmlu_en.csv",
    "/content/mmlu_bn.csv",
    "/content/mmlu_sw.csv",
    "/content/mmlu_te.csv",
    "/content/mmlu_zh.csv",
]

summary_rows = []

for input_file in uploaded_files:
    print("\n" + "=" * 80)
    print("Processing:", input_file)

    lang = infer_lang_from_filename(input_file)
    df = pd.read_csv(input_file)
    validate_input_df(df, input_file)
    df = df[REQUIRED_COLS].copy()

    raw_prompts = [build_raw_prompt(row, lang) for _, row in df.iterrows()]

    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for p in raw_prompts
    ]

    print(f"Generating {len(prompts)} direct responses for {LANG_TO_FULL_NAME[lang]} ({lang})")

    outputs = llm.generate(prompts, sampling_params)

    raw_answers = []
    extracted_answers = []
    parse_methods = []

    for out in outputs:
        text = out.outputs[0].text.strip()
        extracted, method = parse_answer(text, lang)
        raw_answers.append(text)
        extracted_answers.append(extracted)
        parse_methods.append(method)

    df["direct_answer"]    = raw_answers
    df["extracted_answer"] = extracted_answers
    df["parse_method"]     = parse_methods
    df["is_correct"] = (
        df["extracted_answer"].astype(str).str.strip()
        == df["answer"].astype(str).str.strip()
    )

    df = df[[
        "question", "A", "B", "C", "D", "answer",
        "direct_answer", "extracted_answer", "parse_method", "is_correct",
    ]]

    output_file = os.path.join(OUTPUT_DIR, f"direct_{lang}.csv")
    df.to_csv(output_file, index=False, quoting=1)

    correct = int(df["is_correct"].sum())
    total = len(df)
    acc = correct / total if total else 0.0
    parse_counts = df["parse_method"].value_counts().to_dict()

    print(f"Saved: {output_file}")
    print(f"Accuracy: {correct}/{total} = {acc:.1%}")
    print(f"Parse breakdown: {parse_counts}")

    summary_rows.append({
        "language": LANG_TO_FULL_NAME[lang],
        "code": lang,
        "correct": correct,
        "total": total,
        "accuracy": f"{acc:.1%}",
        "parse_failures": int((df["parse_method"] == "failed").sum()),
        "output_file": output_file,
    })

summary_df = pd.DataFrame(summary_rows)
summary_path = os.path.join(OUTPUT_DIR, "summary.csv")
summary_df.to_csv(summary_path, index=False)

print("\n" + "=" * 80)
print("DONE")
print(summary_df.to_string(index=False))
print(f"\nSummary saved to: {summary_path}")
print(f"All outputs are in: {OUTPUT_DIR}")
