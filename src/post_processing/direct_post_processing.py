
"""
Merge direct inference majority outputs into a single Excel workbook.

This script reads only files named `direct_majority_{lang}.*` (CSV or
Excel) from an input folder (default:
`results/direct_inference/mgsm/tiny-aya-global`) and writes a single Excel
workbook with one sheet per input file into the output path (default:
`results/direct_inference/mgsm/final/final_direct_tiny-aya-global.xlsx`).

Unlike the COT post-processing script, this only merges files and does not
compute any `final_answer`/`final_cot` columns.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Set

import pandas as pd


def _read_input_file(path: Path) -> pd.DataFrame:
	"""Read a CSV or Excel file and return a DataFrame.

	For Excel files, try reading the first sheet; if that fails, read all and
	take the first sheet available.
	"""
	if path.suffix.lower() == ".csv":
		return pd.read_csv(path)
	# Excel
	try:
		return pd.read_excel(path, sheet_name=0)
	except Exception:
		xls = pd.read_excel(path, sheet_name=None)
		# take first sheet
		return list(xls.values())[0]


def process_folder(input_folder: str, output_file: str):
	p = Path(input_folder)
	if not p.exists():
		raise FileNotFoundError(f"Input folder not found: {input_folder}")

	files = sorted([
		f for f in p.iterdir()
		if (
			f.is_file()
			and f.suffix.lower() in (".xlsx", ".xls", ".csv")
			and not f.name.startswith("~$")
			and f.name.startswith("direct_majority_")
		)
	])
	if not files:
		raise FileNotFoundError(f"No CSV/XLSX files found in {input_folder}")

	out_parent = Path(output_file).parent
	out_parent.mkdir(parents=True, exist_ok=True)

	writer = pd.ExcelWriter(output_file, engine="openpyxl")
	used_names: Set[str] = set()

	for f in files:
		try:
			df = _read_input_file(f)
		except Exception as e:
			print(f"Warning: failed to read {f}: {e}")
			continue

		# Ensure sheet name uniqueness and length <= 31
		base_name = f.stem[:31]
		sheet_name = base_name
		suffix = 1
		while sheet_name in used_names or len(sheet_name) > 31:
			# reserve room for suffix
			core = base_name[: max(0, 31 - len(str(suffix)) - 1)]
			sheet_name = f"{core}_{suffix}"
			suffix += 1

		used_names.add(sheet_name)
		df.to_excel(writer, sheet_name=sheet_name, index=False)

	writer.close()
	print(f"Merged {len(used_names)} files into {output_file}")


def main():
	ap = argparse.ArgumentParser()
	model_name = 'tiny-aya-global'
	dataset_name = 'mgsm'
	inference_type = 'direct_inference'
	# Resolve defaults relative to the repository root (two parents above `src`)
	repo_root = Path(__file__).resolve().parents[2]
	default_input = str(repo_root / "results" / inference_type / dataset_name / model_name)
	default_output = str(repo_root / "results" / inference_type / dataset_name / "final" / f"final_direct_{model_name}.xlsx")

	ap.add_argument("--input_folder", type=str, default=default_input)
	ap.add_argument("--output_file", type=str, default=default_output)
	args = ap.parse_args()

	input_folder = args.input_folder
	output_file = args.output_file
	if not Path(input_folder).is_absolute():
		input_folder = str(repo_root / input_folder)
	if not Path(output_file).is_absolute():
		output_file = str(repo_root / output_file)

	process_folder(input_folder, output_file)


if __name__ == "__main__":
	main()

