import os
import re
import json
import time
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm
import cohere


SYSTEM_DIRECT = """
You are a precise assistant.
Answer the user's question directly.
Return only the final answer and nothing else.
Do not add explanation.
""".strip()

SYSTEM_COT = """
You are a precise assistant.
Reason step by step, then end with a final line exactly in this format:
Final answer: <answer>
""".strip()


def normalize_whitespace(x: str) -> str:
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_general_answer(x: str) -> str:
    x = normalize_whitespace(x)
    x = x.replace(",", "")
    return x.strip()


def normalize_scan_answer(x: str) -> str:
    return normalize_whitespace(x).upper()


def extract_direct_prediction(raw_text: str) -> str:
    if raw_text is None:
        return ""
    text = str(raw_text).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        text = lines[-1]
    text = re.sub(r"^(Answer|Final answer)\s*:\s*", "", text, flags=re.I).strip()
    return text


def extract_cot_prediction(raw_text: str) -> str:
    if raw_text is None:
        return ""
    text = str(raw_text).strip()

    m = re.search(r"Final answer\s*:\s*(.+)", text, flags=re.I)
    if m:
        return m.group(1).strip()

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        last = lines[-1]
        last = re.sub(r"^(Answer|Final answer)\s*:\s*", "", last, flags=re.I).strip()
        return last

    return text


def score_prediction(bucket: str, gold: str, pred: str) -> bool:
    gold = "" if gold is None else str(gold)
    pred = "" if pred is None else str(pred)

    if bucket == "scan_lite":
        return normalize_scan_answer(gold) == normalize_scan_answer(pred)

    return normalize_general_answer(gold) == normalize_general_answer(pred)


def ask_aya(co_client, model_name: str, question: str, mode: str = "direct") -> str:
    system_msg = SYSTEM_DIRECT if mode == "direct" else SYSTEM_COT

    response = co_client.chat(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": question},
        ],
    )

    return response.message.content[0].text


def ask_aya_with_retry(co_client, model_name: str, question: str, mode: str, max_retries: int = 5, sleep_base: float = 2.0):
    last_err = None
    for attempt in range(max_retries):
        try:
            return ask_aya(co_client, model_name, question, mode=mode), None
        except Exception as e:
            last_err = str(e)
            wait_time = sleep_base * (2 ** attempt)
            time.sleep(wait_time)
    return None, last_err


def compute_group_accuracy(df: pd.DataFrame, group_col: str, correct_col: str) -> pd.DataFrame:
    out = (
        df.groupby(group_col, dropna=False)[correct_col]
        .agg(["sum", "count", "mean"])
        .reset_index()
        .rename(columns={"sum": "num_correct", "count": "num_total", "mean": "accuracy"})
        .sort_values(group_col)
    )
    return out


def build_summary(results_df: pd.DataFrame):
    summary = {}

    def acc(col):
        if len(results_df) == 0:
            return 0.0
        return float(results_df[col].mean())

    summary["num_examples"] = int(len(results_df))
    summary["direct_accuracy"] = acc("direct_correct")
    summary["cot_accuracy"] = acc("cot_correct")
    summary["direct_better_count"] = int(((results_df["direct_correct"] == True) & (results_df["cot_correct"] == False)).sum())
    summary["cot_better_count"] = int(((results_df["direct_correct"] == False) & (results_df["cot_correct"] == True)).sum())
    summary["both_correct_count"] = int(((results_df["direct_correct"] == True) & (results_df["cot_correct"] == True)).sum())
    summary["both_wrong_count"] = int(((results_df["direct_correct"] == False) & (results_df["cot_correct"] == False)).sum())

    return summary


def ensure_columns(df: pd.DataFrame):
    required = ["id", "bucket", "difficulty", "question", "answer"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")


def load_dataset(path: str) -> pd.DataFrame:
    path = str(path)
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
        if "metadata" in df.columns:
            meta_df = pd.json_normalize(df["metadata"]).add_prefix("meta_")
            df = pd.concat([df.drop(columns=["metadata"]), meta_df], axis=1)
    else:
        raise ValueError("Dataset must be .csv or .jsonl")

    ensure_columns(df)
    return df


def maybe_resume(existing_path: Path, df_full: pd.DataFrame):
    if not existing_path.exists():
        return df_full.copy(), pd.DataFrame()

    prev = pd.read_csv(existing_path)
    if "id" not in prev.columns:
        return df_full.copy(), pd.DataFrame()

    done_ids = set(prev["id"].astype(str).tolist())
    remaining = df_full[~df_full["id"].astype(str).isin(done_ids)].copy()
    return remaining, prev


def process_one_row(row, api_key: str, model_name: str):
    co_client = cohere.ClientV2(api_key=api_key)

    qid = str(row["id"])
    bucket = row["bucket"]
    difficulty = row["difficulty"]
    question = str(row["question"])
    gold = str(row["answer"])

    direct_raw, direct_err = ask_aya_with_retry(
        co_client, model_name, question, mode="direct"
    )
    direct_raw = "" if direct_raw is None else direct_raw
    direct_pred = extract_direct_prediction(direct_raw)
    direct_correct = False if direct_err else score_prediction(bucket, gold, direct_pred)

    cot_raw, cot_err = ask_aya_with_retry(
        co_client, model_name, question, mode="cot"
    )
    cot_raw = "" if cot_raw is None else cot_raw
    cot_pred = extract_cot_prediction(cot_raw)
    cot_correct = False if cot_err else score_prediction(bucket, gold, cot_pred)

    return {
        "id": qid,
        "bucket": bucket,
        "difficulty": difficulty,
        "question": question,
        "gold_answer": gold,

        "direct_raw": direct_raw,
        "direct_pred": direct_pred,
        "direct_correct": direct_correct,
        "direct_error": "" if direct_err is None else direct_err,

        "cot_raw": cot_raw,
        "cot_pred": cot_pred,
        "cot_correct": cot_correct,
        "cot_error": "" if cot_err is None else cot_err,

        "same_prediction": normalize_general_answer(direct_pred) == normalize_general_answer(cot_pred),
        "direct_only_win": bool(direct_correct and not cot_correct),
        "cot_only_win": bool(cot_correct and not direct_correct),
        "both_correct": bool(direct_correct and cot_correct),
        "both_wrong": bool((not direct_correct) and (not cot_correct)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./aya_sanity_outputs")
    parser.add_argument("--api_key", type=str, default=None)
    parser.add_argument("--model_name", type=str, default="tiny-aya-global")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=20)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max_workers", type=int, default=3)
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("COHERE_API_KEY")
    if not api_key:
        raise ValueError("COHERE_API_KEY not found. Pass --api_key or set environment variable COHERE_API_KEY.")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.dataset_path)
    if args.limit is not None:
        df = df.head(args.limit).copy()

    detailed_path = output_dir / "detailed_results.csv"

    if args.resume:
        df_remaining, prev_results = maybe_resume(detailed_path, df)
        results = prev_results.to_dict("records")
    else:
        df_remaining = df.copy()
        results = []

    print(f"Loaded dataset: {args.dataset_path}")
    print(f"Total rows selected: {len(df)}")
    print(f"Rows remaining to run: {len(df_remaining)}")
    print(f"Output dir: {output_dir}")
    print(f"Max workers: {args.max_workers}")

    rows = df_remaining.to_dict("records")

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        future_to_id = {
            executor.submit(process_one_row, row, api_key, args.model_name): str(row["id"])
            for row in rows
        }

        for i, future in enumerate(tqdm(as_completed(future_to_id), total=len(future_to_id), desc="Running evaluation")):
            row_id = future_to_id[future]
            try:
                result_row = future.result()
            except Exception as e:
                result_row = {
                    "id": row_id,
                    "bucket": "",
                    "difficulty": "",
                    "question": "",
                    "gold_answer": "",

                    "direct_raw": "",
                    "direct_pred": "",
                    "direct_correct": False,
                    "direct_error": f"Worker failed: {str(e)}",

                    "cot_raw": "",
                    "cot_pred": "",
                    "cot_correct": False,
                    "cot_error": f"Worker failed: {str(e)}",

                    "same_prediction": False,
                    "direct_only_win": False,
                    "cot_only_win": False,
                    "both_correct": False,
                    "both_wrong": True,
                }

            results.append(result_row)

            if ((len(results) % args.save_every) == 0) or (i == len(future_to_id) - 1):
                temp_df = pd.DataFrame(results)
                temp_df.to_csv(detailed_path, index=False)

    results_df = pd.DataFrame(results)

    summary = build_summary(results_df)
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    direct_by_bucket = compute_group_accuracy(results_df, "bucket", "direct_correct").rename(
        columns={"num_correct": "direct_num_correct", "num_total": "num_total", "accuracy": "direct_accuracy"}
    )
    cot_by_bucket = compute_group_accuracy(results_df, "bucket", "cot_correct").rename(
        columns={"num_correct": "cot_num_correct", "num_total": "num_total_2", "accuracy": "cot_accuracy"}
    )
    bucket_compare = direct_by_bucket.merge(cot_by_bucket, on="bucket", how="outer")
    if "num_total_2" in bucket_compare.columns:
        bucket_compare = bucket_compare.drop(columns=["num_total_2"])
    bucket_compare["cot_minus_direct"] = bucket_compare["cot_accuracy"] - bucket_compare["direct_accuracy"]
    bucket_compare.to_csv(output_dir / "bucket_summary.csv", index=False)

    direct_by_diff = compute_group_accuracy(results_df, "difficulty", "direct_correct").rename(
        columns={"num_correct": "direct_num_correct", "num_total": "num_total", "accuracy": "direct_accuracy"}
    )
    cot_by_diff = compute_group_accuracy(results_df, "difficulty", "cot_correct").rename(
        columns={"num_correct": "cot_num_correct", "num_total": "num_total_2", "accuracy": "cot_accuracy"}
    )
    diff_compare = direct_by_diff.merge(cot_by_diff, on="difficulty", how="outer")
    if "num_total_2" in diff_compare.columns:
        diff_compare = diff_compare.drop(columns=["num_total_2"])
    diff_compare["cot_minus_direct"] = diff_compare["cot_accuracy"] - diff_compare["direct_accuracy"]
    diff_compare.to_csv(output_dir / "difficulty_summary.csv", index=False)

    results_df[results_df["direct_only_win"]].to_csv(output_dir / "direct_only_wins.csv", index=False)
    results_df[results_df["cot_only_win"]].to_csv(output_dir / "cot_only_wins.csv", index=False)
    results_df[results_df["both_wrong"]].to_csv(output_dir / "both_wrong.csv", index=False)

    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"Num examples        : {summary['num_examples']}")
    print(f"Direct accuracy     : {summary['direct_accuracy']:.4f}")
    print(f"CoT accuracy        : {summary['cot_accuracy']:.4f}")
    print(f"Direct-only wins    : {summary['direct_better_count']}")
    print(f"CoT-only wins       : {summary['cot_better_count']}")
    print(f"Both correct        : {summary['both_correct_count']}")
    print(f"Both wrong          : {summary['both_wrong_count']}")

    print("\nBucket summary:")
    print(bucket_compare.to_string(index=False))

    print("\nDifficulty summary:")
    print(diff_compare.to_string(index=False))

    print(f"\nSaved to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
