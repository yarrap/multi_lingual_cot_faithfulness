import os
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    raise ValueError("COHERE_API_KEY not found in .env file")

MODEL_NAME = "tiny-aya-global"

ALL_LANGUAGES = ["en", "bn", "sw", "te", "zh"]

LANG_TO_FULL_NAME = {
    "en": "English",
    "bn": "Bengali",
    "sw": "Swahili",
    "te": "Telugu",
    "zh": "Chinese",
}
