import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")

MODEL_NAME = "tiny-aya-global"

# Base Path Logic
CONFIG_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CONFIG_DIR, "..", ".."))

DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets")
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")

# Inference Results Directories
MMLU_INFERENCE_DIR = os.path.join(RESULTS_DIR, "cot_inference", "mmlu", MODEL_NAME)
MGSM_INFERENCE_DIR = os.path.join(RESULTS_DIR, "cot_inference", "mgsm", MODEL_NAME)

# Perturbation Results Directories
MMLU_PERTURBATION_DIR = os.path.join(RESULTS_DIR, "truncation_perturbation", "mmlu", MODEL_NAME)

ALL_LANGUAGES = ["en", "bn", "sw", "te", "zh"]

LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}
