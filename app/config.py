from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def get_setting(name: str, default=None):
    """Read Streamlit Cloud secrets first, then environment/.env."""
    try:
        import streamlit as st
        value = st.secrets.get(name, None)
        if value not in (None, ""):
            return str(value)
    except Exception:
        pass
    value = os.getenv(name)
    return value if value not in (None, "") else default


OCR_DIR = ROOT / "data" / "ocr_pages"
CHUNKS_DIR = ROOT / "data" / "chunks"
VECTOR_DIR = ROOT / "vectorstore"
CHAT_DB_PATH = Path(get_setting("CHAT_DB_PATH", str(ROOT / "data" / "ayurvana_history.sqlite3")))
MODEL_DIR = ROOT / "models"
PLANT_MODEL_PATH = Path(get_setting("PLANT_MODEL_PATH", str(MODEL_DIR / "plant_disease_efficientnet_b0.pth")))
COLLECTION_NAME = get_setting("COLLECTION_NAME", "ashtanga_sangraha_vol1")
EMBEDDING_MODEL = get_setting("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
GROQ_MODEL = get_setting("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_API_KEY = get_setting("GROQ_API_KEY")
TAVILY_API_KEY = get_setting("TAVILY_API_KEY")

for directory in (OCR_DIR, CHUNKS_DIR, VECTOR_DIR, CHAT_DB_PATH.parent, MODEL_DIR):
    directory.mkdir(parents=True, exist_ok=True)
