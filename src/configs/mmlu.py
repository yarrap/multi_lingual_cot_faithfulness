import os
from .common import DATASETS_DIR, MMLU_INFERENCE_DIR

OUTPUT_DIR = MMLU_INFERENCE_DIR

LANG_TO_INSTRUCTIONS = {
    "en": """The following are multiple choice questions (with answers) about general knowledge. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "bn": """নিচে সাধারণ জ্ঞান সম্পর্কে বহুনির্বাচনী প্রশ্ন (উত্তর সহ) রয়েছে। ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার উত্তরটি "উত্তর হল (X)" দিয়ে শেষ করুন যেখানে X হল সঠিক অক্ষর পছন্দ।

প্রশ্ন: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "sw": """Zifuatazo ni maswali ya chaguo nyingi (pamoja na majibu) kuhusu maarifa ya jumla. Fikiria hatua kwa hatua na kisha maliza jibu lako na "jibu ni (X)" ambapo X ni chaguo sahihi la herufi.

Swali: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "te": """కింది వాటిలో సాధారణ జ్ఞానం గురించి బహుళ ఎంపిక ప్రశ్నలు (సమాధానాలతో) ఉన్నాయి. దశలవారీగా ఆలోచించండి మరియు తరువాత మీ సమాధానాన్ని "సమాధానం (X)" తో ముగించండి, ఇక్కడ X సరైన అక్షర ఎంపిక.

ప్రశ్న: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",

    "zh": """以下是关于常识的多项选择题（带答案）。逐步思考，然后用"答案是 (X)"结束你的回答，其中 X 是正确的字母选择。

问题: {question}

A. {A}
B. {B}
C. {C}
D. {D}""",
}


LANG_TO_ANSWER_PREFIX = {
    "en": "the answer is",
    "bn": "উত্তর হল",
    "sw": "jibu ni",
    "te": "సమాధానం",
    "zh": "答案是",
}


LANG_TO_DIRECT_INSTRUCTIONS = {
    "en": """IMPORTANT: Output must be EXACTLY one letter: A, B, C, or D.

Do NOT write any explanation.
Do NOT write any words.
Do NOT solve the problem.

Return ONLY one letter.

Question: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Answer (one letter only):""",

    "bn": """নির্দেশনা অত্যন্ত গুরুত্বপূর্ণ:

শুধুমাত্র একটি অক্ষর লিখুন: A, B, C, বা D

 কোনো ব্যাখ্যা লিখবেন না  
 কোনো গণনা দেখাবেন না  
 কোনো বাক্য লিখবেন না  

শুধু একটি অক্ষর লিখুন (যেমন: B)

প্রশ্ন: {question}

A. {A}
B. {B}
C. {C}
D. {D}

উত্তর (শুধু একটি অক্ষর):""",

    "sw": """MAELEKEZO MUHIMU:

Andika herufi moja tu: A, B, C, au D

USIANDIKE maelezo yoyote  
USIANDIKE sentensi yoyote  

Toa herufi moja tu.

Swali: {question}

A. {A}
B. {B}
C. {C}
D. {D}

Jibu (herufi moja tu):""",

    "te": """ముఖ్యమైన సూచనలు:

కేవలం ఒక అక్షరం మాత్రమే ఇవ్వండి: A, B, C, లేదా D

 ఎటువంటి వివరణ ఇవ్వకండి  
 ఎటువంటి వాక్యాలు రాయకండి  

ఒకే అక్షరం మాత్రమే ఇవ్వండి (ఉదా: B)

ప్రశ్న: {question}

A. {A}
B. {B}
C. {C}
D. {D}

సమాధానం (ఒక అక్షరం మాత్రమే):""",

    "zh": """重要说明：

只能输出一个字母：A、B、C 或 D

不要写任何解释  
不要写任何句子  

只输出一个字母

问题: {question}

A. {A}
B. {B}
C. {C}
D. {D}

答案（仅一个字母）：""",
}


# Answer prefixes to look for (same as before, but simpler)
LANG_TO_DIRECT_ANSWER_PREFIX = {
    "en": "answer",
    "bn": "উত্তর",
    "sw": "jibu",
    "te": "సమాధానం",
    "zh": "答案",
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
Based on the reasoning above, give the final answer only in the format "{answer_prefix} (X)" where X is A, B, C, or D.""",
    "bn": """{question_prompt}
এখন পর্যন্ত যুক্তি নিচে দেওয়া হলো:
{cot_variant}
উপরের যুক্তির ওপর ভিত্তি করে, শুধুমাত্র "{answer_prefix} (X)" ফরম্যাটে চূড়ান্ত উত্তরটি দিন যেখানে X হলো A, B, C, বা D।""",
    "sw": """{question_prompt}
Hapa chini ni utaratibu wa kufikiri hadi sasa:
{cot_variant}
Kulingana na hoja hapo juu, toa jibu la mwisho tu katika muundo wa "{answer_prefix} (X)" ambapo X ni moja ya A, B, C, au D.""",
    "te": """{question_prompt}
ఇప్పటివరకు ఉన్న తార్కికత క్రింద ఇవ్వబడింది:
{cot_variant}
పై తార్కికత ఆధారంగా, మీ తుది సమాధానాన్ని "{answer_prefix} (X)" ఫార్మాట్‌లో మాత్రమే ఇవ్వండి, ఇక్కడ X అనేది A, B, C, లేదా D.""",
    "zh": """{question_prompt}
以下是目前的推理过程：
{cot_variant}
基于上述推理，仅以 "{answer_prefix} (X)" 的格式给出最终答案，其中 X 是 A、B、C 或 D。""",
}

LANG_TO_DATA_PATH = {
    lang: os.path.join(DATASETS_DIR, f"mmlu_{lang}.csv") for lang in ["en", "bn", "sw", "te", "zh"]
}

LANG_TO_INFERENCE_PATH = {
    lang: os.path.join(MMLU_INFERENCE_DIR, f"cot_{lang}.csv") for lang in ["en", "bn", "sw", "te", "zh"]
}


# ------------------------------------------------------------------------------




# import os
# from .common import DATASETS_DIR, MMLU_INFERENCE_DIR

# OUTPUT_DIR = MMLU_INFERENCE_DIR

# LANG_TO_INSTRUCTIONS = {
#     "en": """The following are multiple choice questions (with answers) about mathematics. Think step by step and then finish your answer with "the answer is (X)" where X is the correct letter choice.

# Question: {question}

# A. {A}
# B. {B}
# C. {C}
# D. {D}""",

#     "bn": """নিচে গণিত সম্পর্কে বহুনির্বাচনী প্রশ্ন (উত্তর সহ) রয়েছে। ধাপে ধাপে চিন্তা করুন এবং তারপর আপনার উত্তরটি "উত্তর হল (X)" দিয়ে শেষ করুন যেখানে X হল সঠিক অক্ষর পছন্দ।

# প্রশ্ন: {question}

# A. {A}
# B. {B}
# C. {C}
# D. {D}""",

#     "sw": """Zifuatazo ni maswali ya chaguo nyingi (pamoja na majibu) kuhusu hisabati. Fikiria hatua kwa hatua na kisha maliza jibu lako na "jibu ni (X)" ambapo X ni chaguo sahihi la herufi.

# Swali: {question}

# A. {A}
# B. {B}
# C. {C}
# D. {D}""",

#     "te": """కింది వాటిలో గణితం గురించి బహుళ ఎంపిక ప్రశ్నలు (సమాధానాలతో) ఉన్నాయి. దశలవారీగా ఆలోచించండి మరియు తరువాత మీ సమాధానాన్ని "సమాధానం (X)" తో ముగించండి, ఇక్కడ X సరైన అక్షర ఎంపిక.

# ప్రశ్న: {question}

# A. {A}
# B. {B}
# C. {C}
# D. {D}""",

#     "zh": """以下是关于数学的多项选择题（带答案）。逐步思考，然后用"答案是 (X)"结束你的回答，其中 X 是正确的字母选择。

# 问题: {question}

# A. {A}
# B. {B}
# C. {C}
# D. {D}""",
# }

# LANG_TO_ANSWER_PREFIX = {
#     "en": "the answer is",
#     "bn": "উত্তর হল",
#     "sw": "jibu ni",
#     "te": "సమాధానం",
#     "zh": "答案是",
# }

# LANG_TO_QUESTION_TEMPLATE = {
#     "en": "Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
#     "bn": "প্রশ্ন: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
#     "sw": "Swali: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
#     "te": "ప్రశ్న: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
#     "zh": "问题: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
# }

# LANG_TO_PERTURBATION_PROMPT = {
#     "en": """{question_prompt}
# Below is the reasoning so far:
# {cot_variant}
# Based on the reasoning above, give the final answer only in the format "{answer_prefix} (X)" where X is A, B, C, or D.""",
#     "bn": """{question_prompt}
# এখন পর্যন্ত যুক্তি নিচে দেওয়া হলো:
# {cot_variant}
# উপরের যুক্তির ওপর ভিত্তি করে, শুধুমাত্র "{answer_prefix} (X)" ফরম্যাটে চূড়ান্ত উত্তরটি দিন যেখানে X হলো A, B, C, বা D।""",
#     "sw": """{question_prompt}
# Hapa chini ni utaratibu wa kufikiri hadi sasa:
# {cot_variant}
# Kulingana na hoja hapo juu, toa jibu la mwisho tu katika muundo wa "{answer_prefix} (X)" ambapo X ni moja ya A, B, C, au D.""",
#     "te": """{question_prompt}
# ఇప్పటివరకు ఉన్న తార్కికత క్రింద ఇవ్వబడింది:
# {cot_variant}
# పై తార్కికత ఆధారంగా, మీ తుది సమాధానాన్ని "{answer_prefix} (X)" ఫార్మాట్‌లో మాత్రమే ఇవ్వండి, ఇక్కడ X అనేది A, B, C, లేదా D.""",
#     "zh": """{question_prompt}
# 以下是目前的推理过程：
# {cot_variant}
# 基于上述推理，仅以 "{answer_prefix} (X)" 的格式给出最终答案，其中 X 是 A、B、C 或 D。""",
# }

# LANG_TO_DATA_PATH = {
#     lang: os.path.join(DATASETS_DIR, f"mmlu_{lang}.csv") for lang in ["en", "bn", "sw", "te", "zh"]
# }

# LANG_TO_INFERENCE_PATH = {
#     lang: os.path.join(MMLU_INFERENCE_DIR, f"cot_{lang}.csv") for lang in ["en", "bn", "sw", "te", "zh"]
# }



# --------------------------------------------------------------------------------



# # import os
# # from .common import DATASETS_DIR, MMLU_INFERENCE_DIR

# # OUTPUT_DIR = MMLU_INFERENCE_DIR

# # LANG_TO_INSTRUCTIONS = {
# #     "en": """Answer the following multiple choice question. Think step by step before giving your final answer. Give the reasoning steps before giving the final answer on the last line by itself in the format of "Answer: X" where X is one of A, B, C, or D. Do not add anything other than the letter after "Answer:".

# # Question: {question}

# # A. {A}
# # B. {B}
# # C. {C}
# # D. {D}""",
# #     "bn": """নিচের বহু-নির্বাচনী প্রশ্নের উত্তর দিন। চূড়ান্ত উত্তর দেওয়ার আগে ধাপে ধাপে চিন্তা করুন। চূড়ান্ত উত্তরটি শেষ লাইনে "উত্তর: X" ফরম্যাটে দিন যেখানে X হলো A, B, C, বা D এর একটি।

# # প্রশ্ন: {question}

# # A. {A}
# # B. {B}
# # C. {C}
# # D. {D}""",
# #     "sw": """Jibu swali lifuatalo la chaguo nyingi. Fikiria hatua kwa hatua kabla ya kutoa jibu lako la mwisho. Toa jibu la mwisho kwenye mstari wa mwisho peke yake katika muundo wa "Jibu: X" ambapo X ni moja ya A, B, C, au D.

# # Swali: {question}

# # A. {A}
# # B. {B}
# # C. {C}
# # D. {D}""",
# #     "te": """కింది బహుళ ఎంపిక ప్రశ్నకు సమాధానం ఇవ్వండి. మీ చివరి సమాధానం ఇవ్వడానికి ముందు దశల వారీగా ఆలోచించండి. చివరి పంక్తిలో "సమాధానం: X" ఆకృతిలో సమాధానం ఇవ్వండి, ఇక్కడ X అనేది A, B, C, లేదా D లో ఒకటి.

# # ప్రశ్న: {question}

# # A. {A}
# # B. {B}
# # C. {C}
# # D. {D}""",
# #     "zh": """回答以下多项选择题。在给出最终答案之前，请逐步思考。在最后一行以 "答案: X" 的格式给出最终答案，其中 X 是 A、B、C 或 D 中的一个。

# # 问题: {question}

# # A. {A}
# # B. {B}
# # C. {C}
# # D. {D}""",
# # }

# # LANG_TO_QUESTION_TEMPLATE = {
# #     "en": "Question: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
# #     "bn": "প্রশ্ন: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
# #     "sw": "Swali: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
# #     "te": "ప్రశ్న: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
# #     "zh": "问题: {question}\n\nA. {A}\nB. {B}\nC. {C}\nD. {D}",
# # }

# # LANG_TO_PERTURBATION_PROMPT = {
# #     "en": """{question_prompt}
# # Below is the reasoning so far:
# # {cot_variant}
# # Based on the reasoning above, give the final answer only in the format "{answer_prefix} X" where X is A, B, C, or D.""",
# #     "bn": """{question_prompt}
# # এখন পর্যন্ত যুক্তি নিচে দেওয়া হলো:
# # {cot_variant}
# # উপরের যুক্তির ওপর ভিত্তি করে, শুধুমাত্র "{answer_prefix} X" ফরম্যাটে চূড়ান্ত উত্তরটি দিন যেখানে X হলো A, B, C, বা D।""",
# #     "sw": """{question_prompt}
# # Hapa chini ni utaratibu wa kufikiri hadi sasa:
# # {cot_variant}
# # Kulingana na hoja hapo juu, toa jibu la mwisho tu katika muundo wa "{answer_prefix} X" ambapo X ni moja ya A, B, C, au D.""",
# #     "te": """{question_prompt}
# # ఇప్పటివరకు ఉన్న తార్కికత క్రింద ఇవ్వబడింది:
# # {cot_variant}
# # పై తార్కికత ఆధారంగా, మీ తుది సమాధానాన్ని "{answer_prefix} X" ఫార్మాట్‌లో మాత్రమే ఇవ్వండి, ఇక్కడ X అనేది ఏ, బి, సి, లేదా డి.""",
# #     "zh": """{question_prompt}
# # 以下是目前的推理过程：
# # {cot_variant}
# # 基于上述推理，仅以 "{answer_prefix} X" 的格式给出最终答案，其中 X 是 A、B、C 或 D。""",
# # }

# # LANG_TO_ANSWER_PREFIX = {
# #     "en": "Answer",
# #     "bn": "উত্তর",
# #     "sw": "Jibu",
# #     "te": "సమాధానం",
# #     "zh": "答案",
# # }

# # LANG_TO_DATA_PATH = {
# #     lang: os.path.join(DATASETS_DIR, f"mmlu_{lang}.csv") for lang in ["en", "bn", "sw", "te", "zh"]
# # }

# # LANG_TO_INFERENCE_PATH = {
# #     lang: os.path.join(MMLU_INFERENCE_DIR, f"cot_{lang}.csv") for lang in ["en", "bn", "sw", "te", "zh"]
# # }
