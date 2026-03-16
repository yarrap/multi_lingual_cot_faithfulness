import re
import random
import os
import csv
import glob
from typing import List, Optional


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

class CoTErrorInjector:

    def __init__(self):
        
        # Language-specific configurations based on your setup
        self.lang_config = {
            "en": {"prefix": "Answer", "delims": r'[.!?]'},
            "bn": {"prefix": "উত্তর", "delims": r'[।!?]'},
            "sw": {"prefix": "Jibu", "delims": r'[.!?]'},
            "te": {"prefix": "సమాధానం", "delims": r'[।!?.]'},
            "zh": {"prefix": "答案", "delims": r'[。！？]'},
            "yo": {"prefix": "Idahun", "delims": r'[.!?]'}
        }
        
        # Matches numbers including commas/decimals, but stops before units/text
        # This handles "180 meters", "130,000$", or "60mins"
        self._num_re = re.compile(r'(\d[\d,.]*)', re.UNICODE)
        
        # Pattern for math operations: numbers followed by ops or equals
        self._math_ops_re = re.compile(r'\d\s*[\+\-\*/=<>]|(?<=\s)[\+\-]\s*\d')

    def _get_answer_re(self, lang: str):
        """Dynamically build protection regex for the specific language prefix."""
        prefix = self.lang_config.get(lang, {}).get("prefix", "Answer")
        # Matches the start of a line with the prefix to protect the final label
        return re.compile(rf'^\s*{re.escape(prefix)}\s*[:：]', re.IGNORECASE | re.UNICODE)

    def extract_reasoning_steps(self, cot_trace: str, lang: str = "en") -> List[str]:
        if not cot_trace: return []
        s = cot_trace.strip()
        
        # 1. Split by newlines first (most reliable for CoT steps)
        for sep in ['\n\n', '\n']:
            chunks = [c.strip() for c in s.split(sep) if c.strip()]
            if len(chunks) >= 2: return chunks
        
        # 2. Multilingual sentence splitting fallback
        # Uses specific delimiters (e.g., । for bn/te, 。 for zh)
        delims = self.lang_config.get(lang, {}).get("delims", r'[.!?]')
        sentence_pattern = rf'(?<={delims})\s*'
        return [c.strip() for c in re.split(sentence_pattern, s, flags=re.UNICODE) if c.strip()]

    def _perturb_value(self, num_str: str) -> str:
        """ LOGIC: Applies a significant, plausible-looking error."""
        clean_num = num_str.replace(',', '')
        try:
            if '.' in clean_num:
                val = float(clean_num)
                new_val = random.choice([val + 1.0, val * 1.2, val - 0.5])
                return f"{new_val:.2f}".rstrip('0').rstrip('.')
            else:
                val = int(clean_num)
                choices = [
                    val + random.randint(5, 20),
                    val - random.randint(5, 20),
                    int(val * 1.5),
                    int(val * 0.8)
                ]
                new_val = random.choice(choices)
                if val > 0 and new_val <= 0: new_val = val + 15
                return f"{new_val:,}" if ',' in num_str else str(new_val)
        except:
            return num_str

    def inject_error(self, cot_text: str, lang: str = "en") -> str:
        """Main entry point: Perturbs the last reasoning step before the answer label."""
        if not cot_text or not isinstance(cot_text, str):
            return cot_text

        steps = self.extract_reasoning_steps(cot_text, lang)
        if not steps: return cot_text

        ans_re = self._get_answer_re(lang)

        # 1. Identify math steps that ARE NOT the protected final answer line
        math_indices = [i for i, step in enumerate(steps) 
                        if self._math_ops_re.search(step) and not ans_re.search(step)]

        # 2. Target the final step of reasoning
        if not math_indices:
            target_idx = max(0, len(steps) - 2)
        else:
            target_idx = math_indices[-1]

        sentence = steps[target_idx]

        # 3. Target the result (text after the last '=')
        if '=' in sentence:
            parts = sentence.rsplit('=', 1)
            # Find the first number in the RHS (the result)
            num_match = self._num_re.search(parts[1])
            if num_match:
                corrupted_val = self._perturb_value(num_match.group(1))
                # Reconstruct keeping units (e.g., "180 meters" -> "195 meters")
                new_rhs = parts[1][:num_match.start()] + corrupted_val + parts[1][num_match.end():]
                steps[target_idx] = parts[0] + "=" + new_rhs
            else:
                steps[target_idx] = self._general_perturb(sentence)
        else:
            steps[target_idx] = self._general_perturb(sentence)

        # Join steps with original spacing
        sep = '\n\n' if '\n\n' in cot_text else ('\n' if '\n' in cot_text else ' ')
        return sep.join(steps)

    def _general_perturb(self, text: str) -> str:
        """Fallback: finds the last number in the sentence and perturbs it."""
        matches = list(self._num_re.finditer(text))
        if not matches: return text
        target = matches[-1]
        return text[:target.start()] + self._perturb_value(target.group(1)) + text[target.end():]

    def inject_to_csv(self, input_csv_path: str, output_csv_path: str, lang: str,
                      cot_column: str = "cot_answer", inject_rate: float = 1.0, seed: Optional[int] = None):
        if seed is not None: random.seed(seed)
        err_col = f"{cot_column}_errinj"
        os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)

        with open(input_csv_path, newline='', encoding='utf-8') as fin, \
             open(output_csv_path, 'w', newline='', encoding='utf-8') as fout:
            reader = csv.DictReader(fin)
            fieldnames = list(reader.fieldnames)
            if err_col not in fieldnames: fieldnames.append(err_col)
            writer = csv.DictWriter(fout, fieldnames=fieldnames)
            writer.writeheader()

            for row in reader:
                cot = row.get(cot_column, "") or ""
                row[err_col] = cot
                if cot and random.random() <= inject_rate:
                    row[err_col] = self.inject_error(cot, lang)
                writer.writerow(row)

    def inject_folder(self, results_glob: str, cot_column: str = "cot_answer"):
        """Automatically detects language from filename (e.g., cot_zh.csv)."""
        paths = glob.glob(results_glob, recursive=True)
        for p in paths:
            if not p.lower().endswith('.csv'): continue
            
            # Detect language from filename (e.g., "mgsm_te.csv" -> "te")
            filename = os.path.basename(p)
            lang = "en" # default
            for l in self.lang_config.keys():
                if f"_{l}." in filename:
                    lang = l
                    break
            
            # Logic to change basic_te.csv -> err_inj_te.csv
            # 1. Get the directory path
            # 2. Replace 'basic_' with 'err_inj_' in the filename
            dir_name = os.path.dirname(p)
            new_filename = filename.replace("basic_", "err_inj_")
            out_path = os.path.join(dir_name, new_filename)

            print(f"Processing {lang}: {filename} -> {new_filename}")
            self.inject_to_csv(p, out_path, lang, cot_column=cot_column,inject_rate=1.0, seed=42)
            

# Example usage:
if __name__ == "__main__":
   

    injector = CoTErrorInjector()
    # in_file = "multi_lingual_cot_faithfulness/results/cot_inference/mgsm/tiny-aya-global/basic_te.csv"
    # out_file = "multi_lingual_cot_faithfulness/results/cot_inference/mgsm/tiny-aya-global/basic_te_errinj_te1.csv"
    # injector.inject_to_csv(in_file, out_file, lang="en",cot_column="cot_answer", inject_rate=1.0, seed=42)
    

    # or process folder
    base_path = "multi_lingual_cot_faithfulness/results/cot_inference/mgsm/tiny-aya-global/"
    # This glob targets all files starting with "basic_" and ending in ".csv"
    search_pattern = os.path.join(base_path, "basic_*.csv")
    injector.inject_folder(results_glob=search_pattern,cot_column="cot_answer")