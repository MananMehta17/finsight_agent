"""All settings in one place."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).parent
REPORTS_DIR = ROOT / "data" / "reports"       # put annual report PDFs / txt here
VECTORSTORE_DIR = ROOT / "vectorstore"        # FAISS index is saved here
SENTIMENT_MODEL_DIR = ROOT / "models" / "finsent_model"  # fine tuned model from Colab

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # 384 dimensional
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
TOP_K = 4
MAX_AGENT_STEPS = 12   # safety limit so the agent can never loop forever
