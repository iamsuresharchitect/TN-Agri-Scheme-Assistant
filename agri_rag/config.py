"""Central configuration. Every value can be overridden in the .env file."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


# --- Folders -----------------------------------------------------------------
DATA_DIR = Path(os.getenv("DATA_DIR", BASE_DIR / "data"))
PDF_DIR = DATA_DIR / "pdfs"            # drop scheme PDFs here
CACHE_DIR = DATA_DIR / "cache"         # extracted images, downloads, media
INDEX_DIR = DATA_DIR / "index"         # FAISS index + chunk metadata
for _d in (PDF_DIR, CACHE_DIR, INDEX_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Claude API ----------------------------------------------------------------
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
CLAUDE_MAX_TOKENS = _int("CLAUDE_MAX_TOKENS", 1500)

# --- Embeddings & retrieval -----------------------------------------------------
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-m3")   # multilingual (Tamil + English)
EMBED_DEVICE = os.getenv("EMBED_DEVICE", "") or None    # "cpu", "cuda" or blank for auto
EMBED_BATCH_SIZE = _int("EMBED_BATCH_SIZE", 16)
CHUNK_SIZE = _int("CHUNK_SIZE", 1200)          # characters (~300-400 tokens)
CHUNK_OVERLAP = _int("CHUNK_OVERLAP", 200)
TOP_K = _int("TOP_K", 6)
SIM_THRESHOLD = _float("SIM_THRESHOLD", 0.45)  # cosine similarity; below this = "not found"

# --- PDF processing -------------------------------------------------------------
ENABLE_OCR = _bool("ENABLE_OCR", True)
OCR_LANG = os.getenv("OCR_LANG", "tam+eng")
TESSERACT_CMD = os.getenv("TESSERACT_CMD", "")   # e.g. C:\Program Files\Tesseract-OCR\tesseract.exe
MIN_PAGE_TEXT = _int("MIN_PAGE_TEXT", 40)        # fewer chars than this -> treat page as scanned
MIN_IMAGE_PX = _int("MIN_IMAGE_PX", 120)         # skip icons/logos smaller than this
MIN_OCR_TEXT = _int("MIN_OCR_TEXT", 25)          # ignore OCR results shorter than this

# --- Audio / video --------------------------------------------------------------
ENABLE_MEDIA = _bool("ENABLE_MEDIA", True)
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")    # tiny / base / small / medium / large-v3
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "")   # "ta", "en" or blank = auto-detect
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "auto")

# --- URL scraping ---------------------------------------------------------------
ENABLE_URL_SCRAPING = _bool("ENABLE_URL_SCRAPING", True)
ENABLE_PLAYWRIGHT = _bool("ENABLE_PLAYWRIGHT", True)   # render JS-heavy pages
CRAWL_DEPTH = _int("CRAWL_DEPTH", 1)                   # 1 = only links found in your PDFs
MAX_LINKS_PER_PDF = _int("MAX_LINKS_PER_PDF", 40)
REQUEST_TIMEOUT = _int("REQUEST_TIMEOUT", 30)
REQUEST_DELAY = _float("REQUEST_DELAY", 1.5)           # seconds between hits to the same site
MAX_DOWNLOAD_MB = _int("MAX_DOWNLOAD_MB", 200)
MIN_HTML_TEXT = _int("MIN_HTML_TEXT", 200)             # less text -> retry with Playwright
RESPECT_ROBOTS = _bool("RESPECT_ROBOTS", True)
# Many government sites have expired / misconfigured certificates. When a verified request
# fails, retry that public page without certificate checks. Set false to never do this.
SSL_FALLBACK = _bool("SSL_FALLBACK", True)
USER_AGENT = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36 TNAgriSchemeBot/1.0",
)

# --- LangSmith tracing (optional) -----------------------------------------------
# Tracing is on only when LANGSMITH_TRACING=true AND an API key is set.
LANGSMITH_TRACING = _bool("LANGSMITH_TRACING", False)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "")
LANGSMITH_PROJECT = os.getenv("LANGSMITH_PROJECT", "tn-agri-rag")
os.environ.setdefault("LANGSMITH_PROJECT", LANGSMITH_PROJECT)

# --- UI -------------------------------------------------------------------------
APP_TITLE = os.getenv("APP_TITLE", "TN Agri Scheme Assistant")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "")   # set to protect the upload panel
