"""
Here is the logic for the CoT error injector:


Answer Exclusion: Added a filter to ignore lines starting with "Answer:", "The answer is:", etc., so the corruption only hits the reasoning steps.

Robust Number Regex: Updated to handle numbers with commas (e.g., 130,000) and currency symbols.

Aggressive Perturbation: Replaced with a "Plausible Error" generator that can swap digits, shift decimals, or apply a random percentage offset.

Targeting the Result: In a math string like A + B = C, the logic now specifically tries to target C (the result) rather than A or B.



If the input is:
"James runs 3 sprints * 60 meters = 180 meters. Answer: 180"

Regex Match: self._num_re finds 180 but leaves meters alone.

Perturbation: _perturb_value turns 180 into, say, 195.

Reconstruction: The code stitches it back together: ... = 195 meters.

Final Result:
"James runs 3 sprints * 60 meters = 195 meters. \n Answer: 180"

"""

import os
import random
import re
import pandas as pd
from typing import List, Tuple
import sys

# 1. Calculate the path to the project root (2 levels up from src/inference/)
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

# 2. Add that root to Python's search path
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.configs import (
    mgsm
)

LANG_TO_ANSWER_PREFIX = {
    "en": ["Answer:"],
    "bn": ["উত্তর:"],
    "sw": ["Jibu:"],
    "te": ["సమాధానం:"],
    "zh": ["答案：", "答案:"],
}

class CoTErrorInjector:

    def __init__(self):

        self.lang_config = {
            "en": {"prefix": "Answer", "delims": r'[.!?]'},
            "bn": {"prefix": "উত্তর", "delims": r'[।!?]'},
            "sw": {"prefix": "Jibu", "delims": r'[.!?]'},
            "te": {"prefix": "సమాధానం", "delims": r'[।!?.]'},
            "zh": {"prefix": "答案", "delims": r'[。！？]'},
            "yo": {"prefix": "Idahun", "delims": r'[.!?]'}
        }

        self._num_re = re.compile(r'(\d[\d,.]*)', re.UNICODE)
        self._math_ops_re = re.compile(r'\d\s*[\+\-\*/=<>]|(?<=\s)[\+\-]\s*\d')

    def _get_answer_re(self, lang: str):
        prefix = self.lang_config.get(lang, {}).get("prefix", "Answer")
        return re.compile(rf'^\s*{re.escape(prefix)}\s*[:：]', re.IGNORECASE | re.UNICODE)

    def extract_reasoning_steps(self, cot_trace: str, lang: str) -> List[str]:

        if not cot_trace:
            return []

        s = cot_trace.strip()

        for sep in ['\n\n', '\n']:
            chunks = [c.strip() for c in s.split(sep) if c.strip()]
            if len(chunks) >= 2:
                return chunks

        delims = self.lang_config.get(lang, {}).get("delims", r'[.!?]')
        sentence_pattern = rf'(?<={delims})\s*'

        return [
            c.strip()
            for c in re.split(sentence_pattern, s, flags=re.UNICODE)
            if c.strip()
        ]

    def _perturb_value(self, num_str: str) -> str:

        clean = num_str.replace(',', '')

        try:
            if '.' in clean:
                val = float(clean)
                new_val = random.choice([val + 1.0, val * 1.2, val - 0.5])
                return f"{new_val:.2f}".rstrip('0').rstrip('.')

            else:
                val = int(clean)
                new_val = random.choice([
                    val + random.randint(5, 20),
                    val - random.randint(5, 20),
                    int(val * 1.5),
                    int(val * 0.8)
                ])

                if val > 0 and new_val <= 0:
                    new_val = val + 15

                return str(new_val)

        except:
            return num_str

    def inject_error_with_value(self, cot_text: str, lang: str,seed: Optional[int] = None) -> Tuple[str, str]:
        """
        Returns:
            corrupted_text,
            injected_value (the changed number)
        """

        if seed is not None: random.seed(seed)

        if not cot_text or not isinstance(cot_text, str):
            return cot_text, ""

        steps = self.extract_reasoning_steps(cot_text, lang)
        if not steps:
            return cot_text, ""

        ans_re = self._get_answer_re(lang)

        math_indices = [
            i for i, step in enumerate(steps)
            if self._math_ops_re.search(step)
            and not ans_re.search(step)
        ]

        target_idx = math_indices[-1] if math_indices else max(0, len(steps) - 2)
        sentence = steps[target_idx]

        injected_value = ""

        if '=' in sentence:
            parts = sentence.rsplit('=', 1)
            num_match = self._num_re.search(parts[1])

            if num_match:
                original = num_match.group(1)
                corrupted = self._perturb_value(original)
                injected_value = corrupted

                new_rhs = (
                    parts[1][:num_match.start()]
                    + corrupted
                    + parts[1][num_match.end():]
                )

                steps[target_idx] = parts[0] + "=" + new_rhs

        else:
            matches = list(self._num_re.finditer(sentence))
            if matches:
                target = matches[-1]
                original = target.group(1)
                corrupted = self._perturb_value(original)
                injected_value = corrupted

                steps[target_idx] = (
                    sentence[:target.start()]
                    + corrupted
                    + sentence[target.end():]
                )

        sep = '\n\n' if '\n\n' in cot_text else '\n'
        return sep.join(steps), injected_value
    
    def remove_last_answer_prefix(self,cot_text: str, lang: str) -> str:
        """
        Removes everything starting from the last occurrence of the language's Answer prefix.
        """
        prefixes = LANG_TO_ANSWER_PREFIX[lang]
        for prefix in prefixes:
            if prefix in cot_text:
                cot_text = cot_text.split(prefix)[0].strip()
                break
        return cot_text

       
# ==========================================================
#  MAIN EXCEL PIPELINE
# ==========================================================

def inject_excel_file():

    input_path = "results/cot_inference/mgsm/final/final_data_tiny-aya-global.xlsx"
    output_path = "results/hypothesis_2/error_inj_perturbation/mgsm/final_data_error_inj_tiny-aya-global.xlsx"

    languages = ["en", "bn", "zh", "te", "sw"]

    injector = CoTErrorInjector()

    writer = pd.ExcelWriter(output_path, engine="openpyxl")

    for lang in languages:

        sheet_name = f"cot_majority_{lang}"
        print(f"Processing sheet: {sheet_name}")

        df = pd.read_excel(input_path, sheet_name=sheet_name)

        output_rows = []

        for _, row in df.iterrows():

            cot_text = str(row.get("final_cot", "")) or ""

            corrupt, error_val = injector.inject_error_with_value(
                cot_text,
                lang,
                seed=42
            )
            corrupted= injector.remove_last_answer_prefix(corrupt, lang)
            output_rows.append({
                "question": row.get("question", ""),
                "answer": row.get("answer", ""),
                "final_cot": row.get("final_cot", ""),
                "final_answer": row.get("final_answer", ""),
                "error_inj_cot": corrupted,
                "injected_val": error_val
            })

        out_df = pd.DataFrame(output_rows)
        out_df.to_excel(writer, sheet_name=sheet_name, index=False)

    writer.close()
    print(" Error injection complete. Saved to:")
    print(output_path)


# ==========================================================

if __name__ == "__main__":
    inject_excel_file()