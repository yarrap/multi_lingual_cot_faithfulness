from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import pandas as pd
import torch
from transformers import logging as hf_logging


# ============================================================
# Defaults
# ============================================================

DEFAULT_LANGUAGES = ["en", "zh", "bn", "sw", "te"]

RESOURCE_GROUPS = {
    "en": "high",
    "zh": "high",
    "bn": "medium",
    "sw": "low",
    "te": "low",
}

DIRECT_TEMPLATE = """Solve the following math word problem.
Return only the final numeric answer.

Question:
{question}

Final answer:"""

COT_TEMPLATE = """Solve the following math word problem step by step.
At the end, write exactly:
Final answer: <number>

Question:
{question}
"""

MGSM_RAW_BASE = "https://huggingface.co/datasets/juletxara/mgsm/raw/main"

SUPPORTED_MGSM_LANGS = {
    "en", "bn", "de", "es", "fr", "ja", "ru", "sw", "te", "th", "zh"
}


# ============================================================
# Warning / logging control
# ============================================================

def setup_quiet_mode() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*Both `max_new_tokens`.*and `max_length`.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*Passing `generation_config` together with generation-related arguments.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*The following generation flags are not valid and may be ignored.*",
    )
    warnings.filterwarnings(
        "ignore",
        message=r".*You seem to be using the pipelines sequentially on GPU.*",
    )

    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    hf_logging.set_verbosity_error()


# ============================================================
# Utilities
# ============================================================

def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def clean_str(x: Any) -> str:
    return "" if x is None else str(x).strip()


def safe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)

    s = str(x).strip()
    s = s.replace(",", "")

    if re.fullmatch(r"[-+]?\d+/\d+", s):
        num, den = s.split("/")
        den_val = float(den)
        if den_val == 0:
            return None
        return float(num) / den_val

    try:
        return float(s)
    except ValueError:
        return None


NUMBER_PATTERN = r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?"

FINAL_PATTERNS = [
    rf"final\s+answer\s*[:：]\s*({NUMBER_PATTERN})",
    rf"answer\s*[:：]\s*({NUMBER_PATTERN})",
    rf"####\s*({NUMBER_PATTERN})",
]


def extract_final_number(text: str) -> Optional[float]:
    if text is None:
        return None

    s = str(text).strip()
    if not s:
        return None

    for pattern in FINAL_PATTERNS:
        matches = re.findall(pattern, s, flags=re.IGNORECASE)
        if matches:
            val = safe_float(matches[-1])
            if val is not None:
                return val

    nums = re.findall(NUMBER_PATTERN, s)
    if nums:
        val = safe_float(nums[-1])
        if val is not None:
            return val

    return None


def answers_match(pred: Optional[float], gold: float, tol: float = 1e-6) -> bool:
    if pred is None:
        return False
    return math.isclose(float(pred), float(gold), rel_tol=0.0, abs_tol=tol)


def build_prompt(question: str, mode: str) -> str:
    if mode == "direct":
        return DIRECT_TEMPLATE.format(question=question)
    elif mode == "cot":
        return COT_TEMPLATE.format(question=question)
    raise ValueError(f"Unknown mode: {mode}")


def chunk_list(xs, chunk_size):
    for i in range(0, len(xs), chunk_size):
        yield xs[i : i + chunk_size]


def get_hf_token() -> Optional[str]:
    try:
        from google.colab import userdata
        token = userdata.get("HF_TOKEN")
        if token:
            return token
    except Exception:
        pass

    token = os.environ.get("HF_TOKEN", None)
    if token:
        return token

    token = os.environ.get("HUGGINGFACE_TOKEN", None)
    if token:
        return token

    return None


# ============================================================
# MGSM Loading
# ============================================================

def load_mgsm_language_tsv(language: str) -> pd.DataFrame:
    if language not in SUPPORTED_MGSM_LANGS:
        raise ValueError(
            f"Unsupported MGSM language: {language}. "
            f"Supported: {sorted(SUPPORTED_MGSM_LANGS)}"
        )

    url = f"{MGSM_RAW_BASE}/mgsm_{language}.tsv"

    df = pd.read_csv(
        url,
        sep="\t",
        header=None,
        names=["question", "gold_answer"],
        dtype={"question": "string", "gold_answer": "string"},
        keep_default_na=False,
    )

    if df.shape[1] != 2:
        raise ValueError(
            f"Expected 2 columns from {url}, got shape={df.shape}. "
            "Check whether the remote file format changed."
        )

    df["question"] = df["question"].astype(str).str.strip()
    df["gold_answer"] = df["gold_answer"].apply(safe_float)

    if df["gold_answer"].isna().any():
        bad_rows = df[df["gold_answer"].isna()].head(5)
        raise ValueError(
            "Could not parse some MGSM gold answers into numeric values. "
            f"Example bad rows:\n{bad_rows}"
        )

    df.insert(0, "id", [f"{language}_{i}" for i in range(len(df))])
    df["language"] = language
    df["resource_group"] = RESOURCE_GROUPS.get(language, "unknown")
    df["gold_solution_text"] = ""

    return df[
        ["id", "language", "resource_group", "question", "gold_answer", "gold_solution_text"]
    ].copy()


def build_eval_df(
    languages: List[str],
    max_examples_per_language: Optional[int],
) -> pd.DataFrame:
    dfs = []

    for lang in languages:
        lang_df = load_mgsm_language_tsv(lang)

        if max_examples_per_language is not None:
            lang_df = lang_df.iloc[:max_examples_per_language].copy()

        dfs.append(lang_df)

    if not dfs:
        raise ValueError("No languages selected.")

    df = pd.concat(dfs, ignore_index=True)

    if df.empty:
        raise ValueError("No MGSM examples loaded.")

    return df


# ============================================================
# Existing CSV pair loader for H1
# ============================================================

def _normalize_question(q: Any) -> str:
    return str(q).strip()


def load_h1_from_existing_csvs(
    direct_file: str,
    cot_file: str,
    language: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    direct_df = pd.read_csv(direct_file).copy()
    cot_df = pd.read_csv(cot_file).copy()

    direct_needed = {"question", "answer", "raw_answer", "extracted_answer", "is_correct"}
    cot_needed = {"question", "answer", "cot_answer", "extracted_answer", "is_correct"}

    missing_direct = direct_needed - set(direct_df.columns)
    missing_cot = cot_needed - set(cot_df.columns)

    if missing_direct:
        raise ValueError(f"Direct file missing columns: {sorted(missing_direct)}")
    if missing_cot:
        raise ValueError(f"CoT file missing columns: {sorted(missing_cot)}")

    def to_bool(x):
        s = str(x).strip().lower()
        if s in {"true", "1", "yes"}:
            return True
        if s in {"false", "0", "no"}:
            return False
        return None

    # Prefer row-wise alignment if files are already in the same order
    dq = direct_df["question"].astype(str).str.strip().reset_index(drop=True)
    cq = cot_df["question"].astype(str).str.strip().reset_index(drop=True)

    if len(direct_df) == len(cot_df) and (dq == cq).all():
        merged = pd.DataFrame({
            "question": dq,
            "answer": direct_df["answer"].reset_index(drop=True),
            "raw_answer": direct_df["raw_answer"].reset_index(drop=True),
            "direct_extracted_answer": direct_df["extracted_answer"].reset_index(drop=True),
            "direct_is_correct": direct_df["is_correct"].reset_index(drop=True),
            "cot_answer": cot_df["cot_answer"].reset_index(drop=True),
            "cot_extracted_answer": cot_df["extracted_answer"].reset_index(drop=True),
            "cot_is_correct": cot_df["is_correct"].reset_index(drop=True),
        })
    else:
        direct_df["qkey"] = direct_df["question"].astype(str).str.strip()
        cot_df["qkey"] = cot_df["question"].astype(str).str.strip()

        merged_raw = direct_df.merge(
            cot_df,
            on="qkey",
            suffixes=("_direct", "_cot"),
            how="inner",
        )

        if merged_raw.empty:
            raise ValueError("No rows matched between direct and CoT files.")

        merged = pd.DataFrame({
            "question": merged_raw["question_direct"],
            "answer": merged_raw["answer_direct"],
            "raw_answer": merged_raw["raw_answer"],
            "direct_extracted_answer": merged_raw["extracted_answer_direct"],
            "direct_is_correct": merged_raw["is_correct_direct"],
            "cot_answer": merged_raw["cot_answer"],
            "cot_extracted_answer": merged_raw["extracted_answer_cot"],
            "cot_is_correct": merged_raw["is_correct_cot"],
        })

    merged = merged.reset_index(drop=True)
    merged["gold_answer"] = merged["answer"].apply(safe_float)

    if merged["gold_answer"].isna().any():
        bad_rows = merged[merged["gold_answer"].isna()].head(5)
        raise ValueError(f"Could not parse some gold answers:\n{bad_rows}")

    merged["id"] = [f"{language}_{i}" for i in range(len(merged))]
    merged["language"] = language
    merged["resource_group"] = RESOURCE_GROUPS.get(language, "unknown")

    eval_df = merged[["id", "language", "resource_group", "question", "gold_answer"]].copy()
    eval_df["gold_solution_text"] = ""

    direct_results = pd.DataFrame({
        "id": merged["id"],
        "language": merged["language"],
        "resource_group": merged["resource_group"],
        "mode": "direct",
        "question": merged["question"],
        "gold_answer": merged["gold_answer"],
        "prompt": None,
        "raw_output": merged["raw_answer"],
        "pred_answer": merged["direct_extracted_answer"].apply(safe_float),
        "model_name": "imported_csv",
        "correct": merged["direct_is_correct"].map(to_bool),
    })

    cot_results = pd.DataFrame({
        "id": merged["id"],
        "language": merged["language"],
        "resource_group": merged["resource_group"],
        "mode": "cot",
        "question": merged["question"],
        "gold_answer": merged["gold_answer"],
        "prompt": None,
        "raw_output": merged["cot_answer"],
        "pred_answer": merged["cot_extracted_answer"].apply(safe_float),
        "model_name": "imported_csv",
        "correct": merged["cot_is_correct"].map(to_bool),
    })

    if direct_results["correct"].isna().any():
        raise ValueError("Some direct is_correct values could not be parsed to boolean.")
    if cot_results["correct"].isna().any():
        raise ValueError("Some cot is_correct values could not be parsed to boolean.")

    results_df = pd.concat([direct_results, cot_results], ignore_index=True)
    return eval_df, results_df

# ============================================================
# Optimized batched HF inference
# ============================================================

class HFRunner:
    def __init__(
        self,
        model_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        batch_size: int = 16,
        use_bfloat16: bool = True,
        hf_token: Optional[str] = None,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as e:
            raise ImportError(
                "Install first: pip install transformers accelerate sentencepiece"
            ) from e

        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.batch_size = batch_size

        if torch.cuda.is_available():
            dtype = torch.bfloat16 if use_bfloat16 else torch.float16
        else:
            dtype = torch.float32

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            token=hf_token,
            trust_remote_code=True,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()

    def _prepare_chat_texts(self, prompts: List[str]) -> List[str]:
        texts = []
        has_chat_template = (
            hasattr(self.tokenizer, "apply_chat_template")
            and getattr(self.tokenizer, "chat_template", None) is not None
        )

        if has_chat_template:
            for prompt in prompts:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                texts.append(text)
        else:
            texts = prompts

        return texts

    @torch.inference_mode()
    def generate_batch(self, prompts: List[str]) -> List[str]:
        texts = self._prepare_chat_texts(prompts)

        enc = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        try:
            device = next(self.model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
        except Exception:
            pass

        gen_kwargs = dict(
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            use_cache=True,
        )

        if self.temperature > 0.0:
            gen_kwargs["do_sample"] = True
            gen_kwargs["temperature"] = self.temperature
        else:
            gen_kwargs["do_sample"] = False

        out = self.model.generate(**enc, **gen_kwargs)

        input_len = enc["input_ids"].shape[1]
        gen_only = out[:, input_len:]

        decoded = self.tokenizer.batch_decode(
            gen_only,
            skip_special_tokens=True,
        )

        return [x.strip() for x in decoded]


def run_live_generation(
    eval_df: pd.DataFrame,
    model_name: str,
    max_new_tokens: int,
    temperature: float,
    batch_size: int,
    hf_token: Optional[str] = None,
) -> pd.DataFrame:
    runner = HFRunner(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        batch_size=batch_size,
        use_bfloat16=True,
        hf_token=hf_token,
    )

    records = []

    for mode in ["direct", "cot"]:
        rows = eval_df.to_dict(orient="records")
        prompts = [build_prompt(r["question"], mode) for r in rows]

        total_batches = math.ceil(len(rows) / batch_size)
        for batch_idx, (row_chunk, prompt_chunk) in enumerate(
            zip(chunk_list(rows, batch_size), chunk_list(prompts, batch_size)),
            start=1,
        ):
            print(f"[{mode}] batch {batch_idx}/{total_batches}")

            outputs = runner.generate_batch(prompt_chunk)

            for row, prompt, raw_output in zip(row_chunk, prompt_chunk, outputs):
                pred_answer = extract_final_number(raw_output)
                correct = answers_match(pred_answer, row["gold_answer"])

                records.append({
                    "id": row["id"],
                    "language": row["language"],
                    "resource_group": row["resource_group"],
                    "mode": mode,
                    "question": row["question"],
                    "gold_answer": row["gold_answer"],
                    "prompt": prompt,
                    "raw_output": raw_output,
                    "pred_answer": pred_answer,
                    "correct": correct,
                    "model_name": model_name,
                })

    return pd.DataFrame(records)


# ============================================================
# Trace scoring mode
# ============================================================

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_no} in {path}") from e
    return rows


def score_trace_file(eval_df: pd.DataFrame, trace_file: str) -> pd.DataFrame:
    traces = load_jsonl(trace_file)
    trace_df = pd.DataFrame(traces)

    needed = {"id", "mode", "raw_output"}
    missing = needed - set(trace_df.columns)
    if missing:
        raise ValueError(f"Trace file missing required columns: {sorted(missing)}")

    merged = trace_df.merge(
        eval_df[["id", "language", "resource_group", "question", "gold_answer"]],
        on="id",
        how="left",
        validate="many_to_one",
    )

    if merged["gold_answer"].isna().any():
        bad = merged.loc[merged["gold_answer"].isna(), "id"].tolist()[:10]
        raise ValueError(f"Some trace IDs did not match MGSM eval IDs. Example bad IDs: {bad}")

    merged["pred_answer"] = merged["raw_output"].apply(extract_final_number)
    merged["correct"] = merged.apply(
        lambda r: answers_match(r["pred_answer"], r["gold_answer"]),
        axis=1,
    )

    if "prompt" not in merged.columns:
        merged["prompt"] = None
    if "model_name" not in merged.columns:
        merged["model_name"] = "imported_model"

    return merged[
        [
            "id",
            "language",
            "resource_group",
            "mode",
            "question",
            "gold_answer",
            "prompt",
            "raw_output",
            "pred_answer",
            "correct",
            "model_name",
        ]
    ].copy()


# ============================================================
# H1 summaries
# ============================================================

def build_language_summary(results_df: pd.DataFrame) -> pd.DataFrame:
    acc = (
        results_df.groupby(["language", "resource_group", "mode"], as_index=False)["correct"]
        .mean()
        .rename(columns={"correct": "accuracy"})
    )

    pivot = acc.pivot_table(
        index=["language", "resource_group"],
        columns="mode",
        values="accuracy",
        aggfunc="first",
    ).reset_index()

    if "direct" not in pivot.columns:
        pivot["direct"] = pd.NA
    if "cot" not in pivot.columns:
        pivot["cot"] = pd.NA

    pivot = pivot.rename(columns={"direct": "direct_acc", "cot": "cot_acc"})
    pivot["reasoning_effect"] = pivot["cot_acc"] - pivot["direct_acc"]

    return pivot.sort_values(["resource_group", "language"]).reset_index(drop=True)


def build_group_summary(lang_summary_df: pd.DataFrame) -> pd.DataFrame:
    return (
        lang_summary_df.groupby("resource_group", as_index=False)[["direct_acc", "cot_acc", "reasoning_effect"]]
        .mean()
        .sort_values("resource_group")
        .reset_index(drop=True)
    )


def build_overall_summary(results_df: pd.DataFrame, lang_summary_df: pd.DataFrame) -> Dict[str, Any]:
    return {
        "overall_accuracy_by_mode": results_df.groupby("mode")["correct"].mean().to_dict(),
        "mean_reasoning_effect": float(lang_summary_df["reasoning_effect"].mean()),
        "n_examples": int(results_df["id"].nunique()),
        "n_rows": int(len(results_df)),
        "languages": sorted(results_df["language"].unique().tolist()),
    }


# ============================================================
# Output
# ============================================================

def save_outputs(
    out_dir: str,
    eval_df: pd.DataFrame,
    results_df: pd.DataFrame,
    lang_summary_df: pd.DataFrame,
    group_summary_df: pd.DataFrame,
    overall_summary: Dict[str, Any],
) -> None:
    out = ensure_dir(out_dir)

    eval_df.to_csv(out / "mgsm_eval_examples.csv", index=False)
    results_df.to_csv(out / "per_example_results.csv", index=False)
    lang_summary_df.to_csv(out / "h1_language_summary.csv", index=False)
    group_summary_df.to_csv(out / "h1_resource_group_summary.csv", index=False)

    with open(out / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    plot_df = lang_summary_df.sort_values("reasoning_effect", ascending=False)

    plt.figure(figsize=(10, 5))
    plt.bar(plot_df["language"], plot_df["reasoning_effect"])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Language")
    plt.ylabel("CoT Accuracy - Direct Accuracy")
    plt.title("H1 Reasoning Effect by Language")
    plt.tight_layout()
    plt.savefig(out / "reasoning_effect_by_language.png", dpi=200)
    plt.close()


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--languages", type=str, default=",".join(DEFAULT_LANGUAGES))
    parser.add_argument("--max_examples_per_language", type=int, default=None)
    parser.add_argument("--output_dir", type=str, default="outputs/h1_mgsm_run")

    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--trace_file", type=str, default=None)

    parser.add_argument("--direct_file", type=str, default=None)
    parser.add_argument("--cot_file", type=str, default=None)
    parser.add_argument("--file_language", type=str, default=None)

    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch_size", type=int, default=16)

    return parser.parse_args()


def main():
    setup_quiet_mode()
    args = parse_args()

    csv_mode = args.direct_file is not None or args.cot_file is not None

    if csv_mode:
        if not (args.direct_file and args.cot_file and args.file_language):
            raise ValueError(
                "CSV mode requires --direct_file, --cot_file, and --file_language"
            )
    elif (args.model_name is None) == (args.trace_file is None):
        raise ValueError(
            "Provide exactly one of --model_name or --trace_file, "
            "or use --direct_file + --cot_file + --file_language"
        )

    if args.direct_file is not None:
        print(f"Loading existing H1 CSVs for language: {args.file_language}")
        eval_df, results_df = load_h1_from_existing_csvs(
            direct_file=args.direct_file,
            cot_file=args.cot_file,
            language=args.file_language,
        )
        print(f"Loaded {len(eval_df)} aligned examples from existing files.")
    else:
        languages = [x.strip() for x in args.languages.split(",") if x.strip()]

        print("Loading MGSM...")
        eval_df = build_eval_df(
            languages=languages,
            max_examples_per_language=args.max_examples_per_language,
        )
        print(f"Loaded {len(eval_df)} examples across {eval_df['language'].nunique()} languages.")

        if args.trace_file is not None:
            print(f"Scoring trace file: {args.trace_file}")
            results_df = score_trace_file(eval_df, args.trace_file)
        else:
            hf_token = get_hf_token()
            print(f"Running live generation with model: {args.model_name}")
            print(f"Batch size: {args.batch_size} | max_new_tokens: {args.max_new_tokens}")

            results_df = run_live_generation(
                eval_df=eval_df,
                model_name=args.model_name,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                batch_size=args.batch_size,
                hf_token=hf_token,
            )

    lang_summary_df = build_language_summary(results_df)
    group_summary_df = build_group_summary(lang_summary_df)
    overall_summary = build_overall_summary(results_df, lang_summary_df)

    save_outputs(
        out_dir=args.output_dir,
        eval_df=eval_df,
        results_df=results_df,
        lang_summary_df=lang_summary_df,
        group_summary_df=group_summary_df,
        overall_summary=overall_summary,
    )

    print("\n=== H1 per-language summary ===")
    print(lang_summary_df.to_string(index=False))

    print("\n=== H1 per-resource-group summary ===")
    print(group_summary_df.to_string(index=False))

    print("\n=== Overall summary ===")
    print(json.dumps(overall_summary, indent=2))

    print(f"\nSaved to: {args.output_dir}")


if __name__ == "__main__":
    main()
