"""
Post-processing for tiny-aya-global MGSM outputs (vote-based).

This script expects the per-file Excel sheets to already contain columns
indicating the inference runs and a vote column (e.g. `extracted_run1`,
`cot_run1`, `extracted_run2`, `extracted_run3`, `majority_vote`, `vote_status`).

The script will NOT re-parse the raw COT text; instead it uses the extracted
answer columns and the vote information to select a final answer and its
corresponding COT according to these rules:
  - If `vote_status` is `unanimous`: pick any run (first) that has a non-null
    extracted answer and its `cot_runX` as `final_cot`.
  - If `vote_status` is `majority`: use `majority_vote` as `final_answer`, and
    pick the first run whose `extracted_runX` matches that majority to take its
    `cot_runX` as `final_cot`.
  - Otherwise (no vote / all different): pick `extracted_run1` and `cot_run1`.

It reads all Excel files in the input folder and writes a merged Excel workbook
with one sheet per input file, adding `final_answer`, `final_answer_translated`,
`final_cot`, `chosen_inference_index`, and preserving existing columns.

`final_answer_translated` is a clean copy of `final_answer` with all
native-script digits (Bengali, Telugu, Chinese written-form) converted to
ASCII digits. Use this column for all downstream accuracy and analysis work.
`final_answer` is left untouched as the original extracted value.

Usage:
  python tiny_aya_global_post_processing.py --input_folder results/cot_inference/mgsm/tiny-aya-global \
      --output_file results/cot_inference/mgsm/tiny-aya-global/merged_votes.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ── Native numeral normalization ───────────────────────────────────────────────
# Covers Bengali (০-৯), Telugu (౦-౯), and Chinese written-form (〇-九).
# Swahili and English use ASCII digits already — no entry needed.
# Devanagari intentionally excluded as it is not a target language.
# Applied only to final_answer_translated; final_answer is left as-is.
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


def normalize_numerals(value) -> Optional[str]:
    """
    Convert native-script digits in `value` to ASCII equivalents.
    Handles str, int, float, and None/NaN safely.
    Returns a string with ASCII digits, or None if the input was null.
    """
    if value is None:
        return None
    if isinstance(value, float) and pd.isna(value):
        return None
    return str(value).translate(_NUMERAL_TABLE).strip()


def _first_matching_run_index(df_row: pd.Series, majority_value) -> Optional[int]:
    """Return the 0-based index of the first run whose extracted_run matches majority_value."""
    for i in range(1, 4):
        col = f'extracted_run{i}'
        if col in df_row.index:
            val = df_row.get(col)
            if pd.isna(val):
                continue
            try:
                if str(val).strip() == str(majority_value).strip():
                    return i - 1
            except Exception:
                pass
    return None


def process_dataframe_using_votes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    final_answers = []
    final_cots = []
    chosen_idxs = []

    for _, row in df.iterrows():
        vote_status = None
        if 'vote_status' in row.index:
            vote_status = row.get('vote_status')
        majority_val = row.get('majority_vote') if 'majority_vote' in row.index else None

        def cot_for(i: int) -> Optional[str]:
            col = f'cot_run{i+1}'
            return row.get(col) if col in row.index else None

        def extracted_for(i: int) -> Optional[str]:
            col = f'extracted_run{i+1}'
            return row.get(col) if col in row.index else None

        if vote_status is not None and isinstance(vote_status, str) and vote_status.lower() == 'unanimous':
            # pick first available run
            chosen = None
            for i in range(3):
                ext = extracted_for(i)
                if ext is not None and not (isinstance(ext, float) and pd.isna(ext)):
                    chosen = i
                    break
            if chosen is None:
                final_answers.append(None)
                final_cots.append(None)
                chosen_idxs.append(None)
            else:
                final_answers.append(extracted_for(chosen))
                final_cots.append(cot_for(chosen))
                chosen_idxs.append(chosen)

        elif vote_status is not None and isinstance(vote_status, str) and vote_status.lower() == 'majority':
            # use majority_vote and find a run matching it for its cot
            if majority_val is None or (isinstance(majority_val, float) and pd.isna(majority_val)):
                final_answers.append(extracted_for(0))
                final_cots.append(cot_for(0))
                chosen_idxs.append(0)
            else:
                idx = _first_matching_run_index(row, majority_val)
                if idx is not None:
                    final_answers.append(majority_val)
                    final_cots.append(cot_for(idx))
                    chosen_idxs.append(idx)
                else:
                    # no matching run found; pick first
                    final_answers.append(majority_val)
                    final_cots.append(cot_for(0))
                    chosen_idxs.append(0)

        else:
            # no vote info or all_differ: pick first run
            final_answers.append(extracted_for(0) if 'extracted_run1' in row.index else None)
            final_cots.append(cot_for(0) if 'cot_run1' in row.index else None)
            chosen_idxs.append(0 if 'extracted_run1' in row.index else None)

    df['final_answer'] = final_answers
    df['final_cot'] = final_cots
    df['chosen_inference_index'] = chosen_idxs

    # ── ADDED: final_answer_translated ────────────────────────────────────────
    # Clean copy of final_answer with all native-script digits converted to
    # ASCII. Use this column for all downstream accuracy comparisons and
    # analysis. final_answer is preserved exactly as extracted for reference.
    # ──────────────────────────────────────────────────────────────────────────
    df['final_answer_translated'] = df['final_answer'].apply(normalize_numerals)

    return df


def process_folder(input_folder: str, output_file: str):
    p = Path(input_folder)
    if not p.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    files = sorted([
        f for f in p.iterdir()
        if f.suffix.lower() in ('.xlsx', '.xls', '.csv')
        and not f.name.startswith('~$')
        and f.name.startswith('cot_majority_')
    ])
    if not files:
        raise FileNotFoundError(f"No cot_majority_*.xlsx/.xls/.csv files found in {input_folder}")

    out_parent = Path(output_file).parent
    out_parent.mkdir(parents=True, exist_ok=True)
    writer = pd.ExcelWriter(output_file, engine='openpyxl')

    for f in files:
        try:
            if f.suffix.lower() == '.csv':
                df = pd.read_csv(f)
            else:
                try:
                    df = pd.read_excel(f, sheet_name=0)
                except Exception:
                    xls = pd.read_excel(f, sheet_name=None)
                    df = list(xls.values())[0]
        except Exception as e:
            print(f"Warning: failed to read {f}: {e}")
            continue

        processed = process_dataframe_using_votes(df)
        sheet_name = f.stem[:31]
        processed.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"  ✅ Processed: {f.name} → sheet '{sheet_name}'")

    writer.close()
    print(f"\n✅ Merged {len(files)} files into {output_file}")


def main():
    model_name = 'tiny-aya-water'
    dataset_name = 'mgsm'
    inference_type = 'cot_inference'

    ap = argparse.ArgumentParser()
    repo_root = Path(__file__).resolve().parents[2]
    default_input = str(repo_root / 'results' / inference_type / dataset_name / model_name)
    default_output = str(repo_root / 'results' / inference_type / dataset_name / 'final' / f'final_data_{model_name}.xlsx')
    ap.add_argument('--input_folder', type=str, default=default_input)
    ap.add_argument('--output_file',  type=str, default=default_output)
    args = ap.parse_args()

    input_folder = args.input_folder
    output_file  = args.output_file
    if not Path(input_folder).is_absolute():
        input_folder = str(repo_root / input_folder)
    if not Path(output_file).is_absolute():
        output_file = str(repo_root / output_file)

    process_folder(input_folder, output_file)


if __name__ == '__main__':
    main()