"""
Post-processing for MMLU COT inference outputs (single-run, no voting).

Each input CSV is expected to have columns:
    question, A, B, C, D, answer, subject, cot, extracted, parse_method, correct

The script reads all `cot_*.csv` files in the input folder and writes a single
merged Excel workbook with one sheet per language file.

Usage:
    python mmlu_merge.py --input_folder results/cot_inference/mmlu/<model_name> \
                         --output_file  results/cot_inference/mmlu/final/final_data_<model_name>.xlsx

Arguments can also be left at their defaults — edit the DEFAULT_* constants below
or pass them via the CLI.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# ── Defaults (edit here or pass via CLI) ──────────────────────────────────────
MODEL_NAME   = "tiny-aya-fire"
DATASET_NAME = "mmlu"
INFER_TYPE   = "cot_inference"

# File pattern to match inside the input folder
FILE_GLOB    = "cot_*.csv"
# ──────────────────────────────────────────────────────────────────────────────


def merge_mmlu_folder(input_folder: str, output_file: str) -> None:
    """Read all matching CSVs in *input_folder* and write one Excel sheet each."""
    in_path = Path(input_folder)
    if not in_path.exists():
        raise FileNotFoundError(f"Input folder not found: {input_folder}")

    files = sorted([
        f for f in in_path.glob(FILE_GLOB)
        if not f.name.startswith("~$")
    ])
    if not files:
        raise FileNotFoundError(
            f"No files matching '{FILE_GLOB}' found in {input_folder}"
        )

    out_path = Path(output_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        for f in files:
            try:
                df = pd.read_csv(f)
            except Exception as exc:
                print(f"  ⚠️  Skipped {f.name}: {exc}")
                continue

            # Sheet names are limited to 31 characters in Excel
            sheet_name = f.stem[:31]
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"  ✅ {f.name} → sheet '{sheet_name}'  ({len(df):,} rows)")

    print(f"\n✅ Merged {len(files)} file(s) into: {out_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]

    default_input = str(
        repo_root / "results" / INFER_TYPE / DATASET_NAME / MODEL_NAME
    )
    default_output = str(
        repo_root / "results" / INFER_TYPE / DATASET_NAME / "final"
        / f"final_data_{MODEL_NAME}.xlsx"
    )

    ap = argparse.ArgumentParser(
        description="Merge MMLU single-run COT CSVs into one Excel workbook."
    )
    ap.add_argument(
        "--input_folder",
        type=str,
        default=default_input,
        help=f"Folder containing cot_*.csv files (default: {default_input})",
    )
    ap.add_argument(
        "--output_file",
        type=str,
        default=default_output,
        help=f"Output Excel path (default: {default_output})",
    )
    args = ap.parse_args()

    # Resolve relative paths against repo root
    input_folder = args.input_folder
    output_file  = args.output_file
    if not Path(input_folder).is_absolute():
        input_folder = str(repo_root / input_folder)
    if not Path(output_file).is_absolute():
        output_file = str(repo_root / output_file)

    merge_mmlu_folder(input_folder, output_file)


if __name__ == "__main__":
    main()