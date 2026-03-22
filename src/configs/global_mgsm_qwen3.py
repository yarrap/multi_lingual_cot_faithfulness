import os
from .common import RESULTS_DIR

QWEN3_MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507"
HF_DATASET_NAME = "CohereLabs/global-mgsm"

ALL_GLOBAL_MGSM_LANGUAGES = ["en", "zh", "te", "bn", "sw"]

QWEN3_MGSM_DIRECT_DIR = os.path.join(RESULTS_DIR, "direct_inference", "global_mgsm", "qwen3")
QWEN3_MGSM_COT_DIR    = os.path.join(RESULTS_DIR, "cot_inference",    "global_mgsm", "qwen3")
