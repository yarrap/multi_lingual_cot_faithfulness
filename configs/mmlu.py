LANG_TO_INSTRUCTIONS = {
    "en": """Answer the following multiple choice question. Think step by step before giving your final answer. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer: X" where X is one of A, B, C, or D. Do not add anything other than the letter after "Answer:".

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "bn": """নিচের বহু-নির্বাচনী প্রশ্নের উত্তর দিন। চূড়ান্ত উত্তর দেওয়ার আগে ধাপে ধাপে চিন্তা করুন। চূড়ান্ত উত্তরটি শেষ লাইনে "উত্তর: X" ফরম্যাটে দিন যেখানে X হলো A, B, C, বা D এর একটি।

প্রশ্ন: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "sw": """Jibu swali lifuatalo la chaguo nyingi. Fikiria hatua kwa hatua kabla ya kutoa jibu lako la mwisho. Toa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu: X" ambapo X ni moja ya A, B, C, au D.

Swali: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "te": """కింది బహుళ ఎంపిక ప్రశ్నకు సమాధానం ఇవ్వండి. మీ చివరి సమాధానం ఇవ్వడానికి ముందు దశల వారీగా ఆలోచించండి. చివరి పంక్తిలో "సమాధానం: X" ఆకృతిలో సమాధానం ఇవ్వండి, ఇక్కడ X అనేది A, B, C, లేదా D లో ఒకటి.

ప్రశ్న: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
    "zh": """回答以下多项选择题。在给出最终答案之前，请逐步思考。在最后一行以 "答案: X" 的格式给出最终答案，其中 X 是 A、B、C 或 D 中的一个。

问题: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
}

LANG_TO_QUESTION_TEMPLATE = {
    "en": "Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
    "bn": "প্রশ্ন: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
    "sw": "Swali: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
    "te": "ప్రశ్న: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
    "zh": "问题: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
}

LANG_TO_PERTURBATION_PROMPT = {
    "en": """{question_prompt}
Below is the reasoning so far:
{cot_variant}
Based on the reasoning above, give the final answer only in the format "{answer_prefix} X" where X is A, B, C, or D.""",
    "bn": """{question_prompt}
এখন পর্যন্ত যুক্তি নিচে দেওয়া হলো:
{cot_variant}
উপরের যুক্তির ওপর ভিত্তি করে, শুধুমাত্র "{answer_prefix} X" ফরম্যাটে চূড়ান্ত উত্তরটি দিন যেখানে X হলো A, B, C, বা D।""",
    "sw": """{question_prompt}
Hapa chini ni utaratibu wa kufikiri hadi sasa:
{cot_variant}
Kulingana na hoja hapo juu, toa jibu la mwisho tu katika muundo wa "{answer_prefix} X" ambapo X ni A, B, C, au D.""",
    "te": """{question_prompt}
ఇప్పటివరకు ఉన్న తార్కికత క్రింద ఇవ్వబడింది:
{cot_variant}
పై తార్కికత ఆధారంగా, మీ తుది సమాధానాన్ని "{answer_prefix} X" ఫార్మాట్‌లో మాత్రమే ఇవ్వండి, ఇక్కడ X అనేది A, B, C, లేదా D.""",
    "zh": """{question_prompt}
以下是目前的推理过程：
{cot_variant}
基于上述推理，仅以 "{answer_prefix} X" 的格式给出最终答案，其中 X 是 A、B、C 或 D。""",
}

LANG_TO_ANSWER_PREFIX = {
    "en": "Answer",
    "bn": "উত্তর",
    "sw": "Jibu",
    "te": "సమాధానం",
    "zh": "答案",
}

LANG_TO_DATA_PATH = {
    "en": "./datasets/mmlu_en.csv",
    "bn": "./datasets/mmlu_bn.csv",
    "sw": "./datasets/mmlu_sw.csv",
    "te": "./datasets/mmlu_te.csv",
    "zh": "./datasets/mmlu_zh.csv",
}

LANG_TO_INFERENCE_PATH = {
    "en": "results/cot_inference/mmlu/tiny-aya-global/cot_en.csv",
    "bn": "results/cot_inference/mmlu/tiny-aya-global/cot_bn.csv",
    "sw": "results/cot_inference/mmlu/tiny-aya-global/cot_sw.csv",
    "te": "results/cot_inference/mmlu/tiny-aya-global/cot_te.csv",
    "zh": "results/cot_inference/mmlu/tiny-aya-global/cot_zh.csv",
}
