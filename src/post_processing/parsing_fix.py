"""Diagnostics and simple-fix heuristics for parsing failures in COT outputs.

Usage: run as a script from the repo root. By default it scans
`results/cot_inference/mgsm/gemma3` for files starting with `cot_`.

Functionality:
- For each `cot_*.csv` file counts rows where extracted columns are empty
- Shows per-file and per-language summaries and example rows where parsing
  returned no extracted answer.
- Optionally attempts simple heuristic extraction (last number / 'Answer:' line)
  and reports how many of the previously-missing rows would be recovered.

This is intentionally conservative: it only *reports* diagnostics and
offers suggested heuristic extraction; it does not overwrite any files.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple


EXTRACTED_COL_PATTERN = re.compile(r"\bextracted\b|\bextracted_run\b|^extracted", re.IGNORECASE)


def is_missing(val: str) -> bool:
	if val is None:
		return True
	v = str(val).strip()
	if v == "":
		return True
	if v.lower() in {"none", "nan", "null", "n/a"}:
		return True
	return False


def heuristic_extract(text: str) -> Tuple[str, str]:
	"""Try a few heuristics to pull an answer from a COT string.

	Returns (method_name, extracted_value) where method_name is one of:
	- 'answer_tag' (looks for 'Answer:' / 'Final Answer:')
	- 'last_number' (last numeric token)
	- 'first_number' (first numeric token)
	- '' when nothing found
	"""
	if not text:
		return "", ""
	# 1) Answer: or Final answer: lines
	m = re.search(r"(?:Answer|Final answer|Final Answer|Answer:)[:\s]*([\-+]?\$?\d+[\d,]*\.?\d*)", text, re.IGNORECASE)
	if m:
		val = m.group(1).replace("$", "").replace(",", "")
		return "answer_tag", val

	# 2) last numeric token (integers or decimals, possibly negative)
	nums = re.findall(r"[\-+]?\d+(?:\.\d+)?", text.replace(',', ''))
	if nums:
		return "last_number", nums[-1]

	# 3) fallback: nothing
	return "", ""


def analyze_file(path: str, attempt_fix: bool = False, examples: int = 3) -> Dict:
	rows = []
	with open(path, newline='', encoding='utf-8') as fh:
		reader = csv.DictReader(fh)
		for row in reader:
			rows.append(row)

	if not rows:
		return {"path": path, "total_rows": 0, "extracted_columns": {}, "examples": []}

	# find extracted columns
	cols = list(rows[0].keys())
	extracted_cols = [c for c in cols if EXTRACTED_COL_PATTERN.search(c)]

	summary = {"path": path, "total_rows": len(rows), "extracted_columns": {}, "examples": [], "missing_rows": {}}

	for col in extracted_cols:
		missing_indices = []
		for i, row in enumerate(rows):
			if is_missing(row.get(col)):
				missing_indices.append(i)

		summary["extracted_columns"][col] = {
			"missing_count": len(missing_indices),
			"missing_frac": len(missing_indices) / max(1, len(rows)),
			"examples": []
		}

		# capture example rows (question, cot text, parse_method if present)
		for idx in missing_indices[:examples]:
			r = rows[idx]
			ex = {
				"row_index": idx,
				"question": r.get("question"),
				"cot": r.get(col.replace("extracted", "cot"), r.get("cot_run1")),
				"parse_method": r.get(col.replace("extracted", "parse_method")),
			}
			summary["extracted_columns"][col]["examples"].append(ex)

		# store full missing rows for optional export
		summary["missing_rows"][col] = [rows[i] for i in missing_indices]

		# attempt fix heuristics if requested
		if attempt_fix and missing_indices:
			recovered = 0
			recover_examples = []
			for idx in missing_indices:
				r = rows[idx]
				# prefer the corresponding cot_run text if present
				cot_col = col.replace("extracted", "cot")
				cot_text = r.get(cot_col) or r.get("cot_run1") or ""
				method, val = heuristic_extract(cot_text)
				if val:
					recovered += 1
					if len(recover_examples) < examples:
						recover_examples.append({"row_index": idx, "method": method, "value": val})

			summary["extracted_columns"][col]["heuristic_recovered"] = recovered
			summary["extracted_columns"][col]["heuristic_recover_frac"] = recovered / max(1, len(rows))
			summary["extracted_columns"][col]["heuristic_examples"] = recover_examples

	return summary


def scan_dir(directory: str, attempt_fix: bool = False, examples: int = 3) -> Dict:
	results = {}
	files = sorted(f for f in os.listdir(directory) if f.startswith("cot_") and f.endswith(".csv"))
	for fn in files:
		path = os.path.join(directory, fn)
		results[fn] = analyze_file(path, attempt_fix=attempt_fix, examples=examples)
	return results


def write_summary_csv(results: Dict, out_path: str) -> None:
	fields = ["file", "extracted_column", "total_rows", "missing_count", "missing_frac", "heuristic_recovered", "heuristic_recover_frac"]
	with open(out_path, "w", newline="", encoding="utf-8") as fh:
		w = csv.DictWriter(fh, fieldnames=fields)
		w.writeheader()
		for fn, info in results.items():
			total = info.get("total_rows", 0)
			for col, cinfo in info.get("extracted_columns", {}).items():
				w.writerow({
					"file": fn,
					"extracted_column": col,
					"total_rows": total,
					"missing_count": cinfo.get("missing_count", 0),
					"missing_frac": f"{cinfo.get('missing_frac', 0.0):.6f}",
					"heuristic_recovered": cinfo.get("heuristic_recovered", 0),
					"heuristic_recover_frac": f"{cinfo.get('heuristic_recover_frac', 0.0):.6f}",
				})


def export_missing_rows(results: Dict, out_dir: str) -> None:
	os.makedirs(out_dir, exist_ok=True)
	for fn, info in results.items():
		base = os.path.splitext(fn)[0]
		# for each extracted column, dump rows that had missing extracted answers
		for col, rows in info.get("missing_rows", {}).items():
			if not rows:
				continue
			out_fn = f"{base}__{col}__missing.csv"
			out_path = os.path.join(out_dir, out_fn)
			# write all columns present in first row
			with open(out_path, "w", newline="", encoding="utf-8") as fh:
				fieldnames = list(rows[0].keys())
				w = csv.DictWriter(fh, fieldnames=fieldnames)
				w.writeheader()
				for r in rows:
					w.writerow(r)


def main(argv: List[str] | None = None) -> int:
	p = argparse.ArgumentParser(description="Detect parsing failures and try simple heuristics")
	p.add_argument("--dir", default="results/cot_inference/mgsm/gemma3", help="Directory with cot_*.csv files")
	p.add_argument("--attempt-fix", action="store_true", help="Try simple heuristic extraction and report recovery rate")
	p.add_argument("--examples", type=int, default=3, help="Number of example rows to show per column")
	p.add_argument("--out-json", default=None, help="Write detailed JSON report to this path")
	p.add_argument("--summary-csv", default=None, help="Write a compact summary CSV (file, extracted_column, total_rows, missing_count)")
	p.add_argument("--export-missing", default=None, help="Directory to write per-file CSVs of rows missing extracted answers")
	args = p.parse_args(argv)

	directory = args.dir
	if not os.path.isdir(directory):
		print(f"Directory not found: {directory}")
		return 2

	results = scan_dir(directory, attempt_fix=args.attempt_fix, examples=args.examples)

	# Print concise summary
	for fn, info in results.items():
		print(f"FILE: {fn} — rows: {info['total_rows']}")
		for col, cinfo in info["extracted_columns"].items():
			miss = cinfo.get("missing_count", 0)
			frac = cinfo.get("missing_frac", 0.0)
			line = f"  {col}: missing {miss} ({frac:.1%})"
			if args.attempt_fix:
				rec = cinfo.get("heuristic_recovered", 0)
				line += f", heuristically recovered {rec}"
			print(line)
			for ex in cinfo.get("examples", []):
				q = ex.get("question")
				pm = ex.get("parse_method")
				print(f"    example row {ex['row_index']}: parse_method={pm} question={q!r}")

	if args.out_json:
		with open(args.out_json, "w", encoding="utf-8") as fh:
			json.dump(results, fh, indent=2, ensure_ascii=False)
		print(f"Wrote JSON report to {args.out_json}")

	if args.summary_csv:
		write_summary_csv(results, args.summary_csv)
		print(f"Wrote summary CSV to {args.summary_csv}")

	if args.export_missing:
		export_missing_rows(results, args.export_missing)
		print(f"Exported missing-row CSVs to {args.export_missing}")

	return 0


if __name__ == "__main__":
	raise SystemExit(main())

