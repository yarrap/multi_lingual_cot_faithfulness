"""Repair missing parsing results (Qwen3-specific heuristics).

Reads CSVs from a `--src` folder (e.g. the `old` folder), attempts improved
parsing only for rows where all `extracted_*` columns are missing, fills the
first missing `extracted_runX` with the recovered value and sets the
corresponding `parse_method_runX`, then recomputes `majority_vote` and
`vote_status` and writes `cot_majority_<lang>.csv` into the destination.

This version contains additional regex patterns tuned for Qwen failure modes
observed in `results/tests/parsing_report_mmlu_qwen3.json` (more answer tags,
currency, percentages, fractions, "Option X" and trailing-line answers).

Usage:
  python3 src/post_processing/repair_missing_parses_qwen.py --src <old_dir> --dst <out_dir>
"""

from __future__ import annotations

import argparse
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


# Precompiled regexes for Qwen-specific patterns
LETTER_RE = re.compile(r"\b(?:Option|option|Ans|Answer|Final answer|উত্তর)[:\s\-–—]*\(?([A-D])\)?\b", re.IGNORECASE)
OPTION_WORD_RE = re.compile(r"\boption\s*[:\s]*([A-D])\b", re.IGNORECASE)
LETTER_EDGE_RE = re.compile(r"\b([A-D])\b")
NUMBER_RE = re.compile(r"[\-+]?\$?\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?%?")
FRACTION_RE = re.compile(r"\b\d+\/\d+\b")


def improved_extract_qwen(text: str) -> Tuple[str, str]:
    """Try a set of candidate extraction heuristics and return (method, value).
    Returns ('', '') when nothing obvious is found."""
    if not isinstance(text, str) or not text.strip():
        return "", ""
    t = text.strip()

    # 1) Explicit answer tags (English, Bengali, variants)
    m = LETTER_RE.search(t)
    if m:
        return "tag_letter", m.group(1)

    # 2) "Option C" style
    m = OPTION_WORD_RE.search(t)
    if m:
        return "option_word", m.group(1)

    # 3) Look for explicit numeric/monetary patterns after answer tags
    m = re.search(r"(?:Answer|Final answer|উত্তর|Ans)[:\s\-–—]*([\-+]?\$?\d{1,3}(?:[,\d]{0,})?(?:\.\d+)?%?)", t, re.IGNORECASE)
    if m:
        val = m.group(1).replace(',', '')
        return "tag_number", val

    # 4) Fraction explicit
    m = FRACTION_RE.search(t)
    if m:
        return "fraction", m.group(0)

    # 5) Last numeric/currency token in text (handles $2.99, 20, 3.5, 1,234 etc.)
    nums = NUMBER_RE.findall(t.replace('\u200b', ''))
    if nums:
        last = nums[-1].replace(',', '')
        return "last_number", last

    # 6) Last letter A-D near end of text (useful for MCQ final line)
    tail = t[-160:]
    m = LETTER_EDGE_RE.search(tail)
    if m:
        return "last_letter", m.group(1)

    # 7) If the COT ends with a short line (e.g., "Answer: C" missing tag), try last non-empty line
    lines = [ln.strip() for ln in t.splitlines() if ln.strip()]
    if lines:
        last_line = lines[-1]
        # single-letter line
        if len(last_line) in (1, 2) and re.fullmatch(r"\(?[A-D]\)?\.?", last_line, re.IGNORECASE):
            m = re.search(r"([A-D])", last_line, re.IGNORECASE)
            if m:
                return "final_line_letter", m.group(1)
        # single token numeric
        tok = last_line.strip()
        if re.fullmatch(r"[\-+]?\$?\d[\d,]*(?:\.\d+)?%?", tok):
            return "final_line_number", tok.replace(',', '')

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


def repair_file_qwen(src_path: Path, dst_path: Path) -> Dict:
    df = pd.read_csv(src_path)
    extracted_cols = [c for c in df.columns if c.lower().startswith("extracted")]
    if not extracted_cols:
        return {"file": src_path.name, "updated": 0, "total_missing": 0}

    updated = 0
    total_missing = 0
    parse_methods = {}

    for idx, row in df.iterrows():
        if all(is_missing(row.get(c)) for c in extracted_cols):
            total_missing += 1
            # prefer cot_run1..3
            cot_text = ""
            for i in range(1, 4):
                cc = f"cot_run{i}"
                if cc in df.columns and not is_missing(row.get(cc)):
                    cot_text = row.get(cc)
                    break
            if not cot_text and "cot" in df.columns:
                cot_text = row.get("cot")

            method, val = improved_extract_qwen(cot_text or "")
            if val:
                for c in extracted_cols:
                    if is_missing(df.at[idx, c]):
                        df.at[idx, c] = val
                        pm_col = c.replace("extracted", "parse_method")
                        df.at[idx, pm_col] = method
                        updated += 1
                        parse_methods.setdefault(method, 0)
                        parse_methods[method] += 1
                        break

    # recompute majority_vote and vote_status
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

    for f in files:
        stem = f.stem
        if stem.startswith("cot_"):
            lang = stem.split("cot_")[1]
            new_name = f"cot_majority_{lang}.csv"
        else:
            new_name = stem + "_majority.csv"
        dst_path = dst / new_name
        r = repair_file_qwen(f, dst_path)
        print(f"Processed {f.name}: updated {r['updated']} rows, total_missing {r['total_missing']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
