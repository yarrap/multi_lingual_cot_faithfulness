import os
from .common import DATASETS_DIR, MGSM_INFERENCE_DIR

OUTPUT_DIR = MGSM_INFERENCE_DIR

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
    "en": "Answer:",
    "bn": "উত্তর:",
    "sw": "Jibu:",
    "te": "సమాధానం:",
    "zh": "答案:",
    "yo": "Idahun:"
}

LANG_TO_DATA_PATH = {
    lang: os.path.join(DATASETS_DIR, f"mgsm_{lang}.csv") for lang in ["en","bn", "sw", "te", "zh"]
}

LANG_TO_INFERENCE_PATH = {
    lang: os.path.join(MGSM_INFERENCE_DIR, f"cot_{lang}.csv") for lang in ["en","bn", "sw", "te", "zh"]
}
