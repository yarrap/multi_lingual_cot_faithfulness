"""Repair missing parsing results for MMLU gemma3 outputs.

This script reads CSV files from a source directory (e.g. the `old` folder),
attempts improved parsing only for rows where all `extracted_*` columns are
missing, fills the first missing `extracted_runX` with the recovered value and
sets the corresponding `parse_method_runX`, then computes a simple
`majority_vote` and `vote_status` and writes the repaired file as
`cot_majority_<lang>.csv` into the destination directory.

Usage (from repo root):
python3 src/post_processing/repair_missing_parses.py \
  --src results/cot_inference/mmlu/gemma3/old \
  --dst results/cot_inference/mmlu/gemma3
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


def is_missing(val: object) -> bool:
    if val is None:
        return True
    s = str(val).strip()
    if s == "":
        return True
    if s.lower() in {"none", "nan", "null", "n/a"}:
        return True
    return False


LETTER_RE_BN = re.compile(r"উত্তর\s*(?:হল)?\s*[:\(\-]*\s*([A-D])\b", re.IGNORECASE)
LETTER_RE_EN = re.compile(r"\bAnswer[:\s\-]*\(?([A-D])\)?\b", re.IGNORECASE)
LETTER_SIMPLE = re.compile(r"\b([A-D])\b")
NUMBER_RE = re.compile(r"[\-+]?\d+[\d,]*\.?\d*")


def improved_extract(text: str) -> Tuple[str, str]:
    """Return (method, value) or ('', '') if nothing extracted."""
    if not isinstance(text, str) or not text.strip():
        return "", ""
    t = text.strip()
    # Prefer explicit 'উত্তর হল (C)' or 'Answer: C' patterns
    m = LETTER_RE_BN.search(t)
    if m:
        return "bn_letter", m.group(1)
    m = LETTER_RE_EN.search(t)
    if m:
        return "en_letter", m.group(1)

    # Look for phrases like 'উত্তর হল (C)' in Bengali with different spacing
    m = re.search(r"উত্তর\s*হল[:\s\(]*([A-D])", t)
    if m:
        return "bn_letter2", m.group(1)

    # If options A-D present earlier, try to find a letter near end of text
    # look at last 120 chars for a single A-D
    tail = t[-120:]
    m = re.search(r"\b([A-D])\b", tail)
    if m:
        return "last_letter", m.group(1)

    # Numeric answers: prefer explicit 'Answer:' numeric
    m = re.search(r"(?:Answer|Final answer|উত্তর)[:\s]*([\-+]?\$?\d+[\d,]*\.?\d*)", t, re.IGNORECASE)
    if m:
        val = m.group(1).replace("$", "").replace(",", "")
        return "answer_tag", val

    # fallback: last numeric token
    nums = NUMBER_RE.findall(t.replace(',', ''))
    if nums:
        return "last_number", nums[-1]

    return "", ""


def compute_majority_and_status(row: Dict) -> Tuple[Optional[str], Optional[str]]:
    vals = []
    for i in range(1, 4):
        col = f"extracted_run{i}"
        v = row.get(col)
        if not is_missing(v):
            vals.append(str(v).strip())
    if not vals:
        return None, "no_answer"
    cnt = Counter(vals)
    most_common, count = cnt.most_common(1)[0]
    if len(cnt) == 1:
        return most_common, "unanimous"
    total = sum(cnt.values())
    if count > total / 2:
        return most_common, "majority"
    return most_common, "all_differ"


def repair_file(src_path: Path, dst_path: Path) -> Dict:
    df = pd.read_csv(src_path)
    # find extracted cols
    extracted_cols = [c for c in df.columns if c.lower().startswith("extracted")]
    if not extracted_cols:
        return {"file": src_path.name, "updated": 0, "total_missing": 0}

    updated = 0
    total_missing = 0
    parse_methods = {}

    for idx, row in df.iterrows():
        # check if all extracted cols missing
        if all(is_missing(row.get(c)) for c in extracted_cols):
            total_missing += 1
            # get best cot text (prefer cot_run1/2/3)
            cot_text = ""
            for i in range(1, 4):
                cc = f"cot_run{i}"
                if cc in df.columns and not is_missing(row.get(cc)):
                    cot_text = row.get(cc)
                    break
            if not cot_text and "cot" in df.columns:
                cot_text = row.get("cot")

            method, val = improved_extract(cot_text or "")
            if val:
                # fill into first extracted column
                for c in extracted_cols:
                    if is_missing(df.at[idx, c]):
                        df.at[idx, c] = val
                        # set parse_method column if exists, else create
                        pm_col = c.replace("extracted", "parse_method")
                        df.at[idx, pm_col] = method
                        updated += 1
                        parse_methods.setdefault(method, 0)
                        parse_methods[method] += 1
                        break

    # compute majority_vote and vote_status columns
    maj_votes = []
    vote_statuses = []
    for _, row in df.iterrows():
        maj, status = compute_majority_and_status(row)
        maj_votes.append(maj if maj is not None else "")
        vote_statuses.append(status if status is not None else "no_answer")
    df["majority_vote"] = maj_votes
    df["vote_status"] = vote_statuses

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dst_path, index=False)
    return {"file": src_path.name, "updated": updated, "total_missing": total_missing, "methods": parse_methods}


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="Source folder with old CSVs")
    p.add_argument("--dst", required=True, help="Destination folder for repaired files")
    args = p.parse_args(argv)

    src = Path(args.src)
    dst = Path(args.dst)
    if not src.exists():
        print(f"Source folder not found: {src}")
        return 2

    files = sorted(f for f in src.iterdir() if f.suffix.lower() == ".csv" and f.name.startswith("cot_"))
    if not files:
        print("No cot_*.csv files found in source folder")
        return 0

    results = []
    for f in files:
        # build destination filename: insert _majority after 'cot' prefix: cot_bn.csv -> cot_majority_bn.csv
        name = f.name
        parts = name.split(".")
        stem = parts[0]
        rest = ".".join(parts[1:])
        # stem like 'cot_bn' -> insert '_majority' after 'cot'
        if stem.startswith("cot_"):
            lang = stem.split("cot_")[1]
            new_name = f"cot_majority_{lang}.csv"
        else:
            new_name = stem + "_majority.csv"
        dst_path = dst / new_name
        r = repair_file(f, dst_path)
        print(f"Processed {f.name}: updated {r['updated']} rows, total_missing {r['total_missing']}")
        results.append(r)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())