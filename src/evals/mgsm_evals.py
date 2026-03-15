"""
MGSM: Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems. 
Language Models are Multilingual Chain-of-Thought Reasoners
Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, Jason Wei
https://arxiv.org/abs/2210.03057
"""

import sys
import os
import re
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from configs import ALL_LANGUAGES, mgsm


def parse_answer(answer: str, answer_prefix: str) -> str:
    if answer_prefix not in answer:
        return ""

    answer_text = answer.split(answer_prefix)[-1].strip()

    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # return the first number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""


def get_lang_samples(lang: str) -> list[dict[str, str]]:
    path = mgsm.LANG_TO_DATA_PATH[lang]
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for sample in reader:
            samples.append({"inputs": sample['question'], "targets": sample['answer'], "lang": lang})
    return samples


def get_all_samples() -> list[dict[str, str]]:
    examples = []
    for lang in ALL_LANGUAGES:
        examples += get_lang_samples(lang)
    return examples


class MGSMEval():
    # TODO
    pass
