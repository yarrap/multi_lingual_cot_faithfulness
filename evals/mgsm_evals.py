"""
MGSM: Multilingual Grade School Math Benchmark (MGSM) is a benchmark of grade-school math problems. 
Language Models are Multilingual Chain-of-Thought Reasoners
Freda Shi, Mirac Suzgun, Markus Freitag, Xuezhi Wang, Suraj Srivats, Soroush Vosoughi, Hyung Won Chung, Yi Tay, Sebastian Ruder, Denny Zhou, Dipanjan Das, Jason Wei
https://arxiv.org/abs/2210.03057
"""

import re
import csv

ALL_LANGUAGES = ["en", "bn","sw", "te", "zh"]

LANG_TO_PATH = {
    "en": "datasets/mgsm_en.csv",
    "bn": "datasets/mgsm_bn.csv",
    "sw": "datasets/mgsm_sw.csv",
    "te": "datasets/mgsm_te.csv",
    "zh": "datasets/mgsm_zh.csv",
}

LANG_TO_INSTRUCTIONS = {
    "en": """Solve this math problem. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer:". Do not add anything other than the integer answer after "Answer:".

{input}""",
    "bn": """এই গণিতের সমস্যাটি সমাধান করুন। চূড়ান্ত উত্তর দেওয়ার আগে যুক্তিসম্পন্ন পদক্ষেপ প্রদান করুন। চূড়ান্ত উত্তরটি একক সংখ্যা হিসাবে "উত্তর:" এর পরে শেষ লাইনে দিন। "উত্তর:" এর পরে অন্য কিছু যুক্ত করবেন না।.

{input}""",
    "sw": """Suluhisha tatizo hili la hesabu. Toa hatua za mantiki kabla ya kutoa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu:". Usiongeze chochote kingine isipokuwa jibu la integer baada ya "Jibu:".

{input}""",
    "te": """ఈ గణిత సమస్యను పరిష్కరించండి. చివరి సమాధానాన్ని ఇవ్వదానికి ముందు తర్కాత్మక అదుగులను ఇవ్వండి. చివరి పంక్తిలో మాత్రమే 'సమాధానం:' అనే ఆకారంలో చివరి సమాధానాద్ని ఇవ్వండి సమాధానం: తర్వాత పూర్ణాంక సమాధానానికి తప్పించి ఎదేనా చేర్చవద్దు.

{input}""",
    "zh": """解决这个数学问题。在最后一行给出答案前，请提供推理步骤。最后一行应该以 "答案: " 的形式独立给出答案。在 "答案：" 后不要添加除整数答案之外的任何内容。

{input}""",
    "yo": """Yanju iṣoro iṣiro yii. Fun awọn igbesẹ ero ṣaaju fifun idahun ikẹhin lori laini ti o kẹhin funrararẹ ni ọna kika "Idahun:". Maṣe ṣafikun ohunkohun miiran ju idahun odidi odidi lẹhin "Idahun:".
    
{input}"""
}

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত্তর",
    "sw": "Jibu",
    "te": "సమాధానం",
    "zh": "答案",
    "yo": "Idahun"
}


def parse_answer(answer: str, answer_prefix: str) -> str:
    if answer_prefix not in answer:
        return ""

    answer_text = answer.split(answer_prefix)[-1].strip()

    # find all the numbers (including decimals) in the string
    numbers = re.findall(r"\d+\.?\d*", answer_text.replace(",", ""))

    # return the first number (removing trailing decimal point if present),
    # or an empty string if there were no numbers
    return numbers[-1].rstrip(".") if numbers else ""


def get_lang_samples(lang: str) -> list[dict[str, str, str]]:
    path = LANG_TO_PATH[lang]
    samples = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for sample in reader:
            samples.append({"inputs": sample['question'], "targets": sample['answer'], "lang": lang})
    return samples


def get_all_samples() -> list[dict[str, str, str]]:
    examples = []
    for lang in ALL_LANGUAGES:
        examples += get_lang_samples(lang)
    return examples


class MGSMEval():
    # TODO
    pass
