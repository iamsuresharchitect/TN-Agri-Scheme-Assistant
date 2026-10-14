"""OCR for images and speech-to-text for audio/video.

Both are optional: if Tesseract or faster-whisper is not installed, the
functions log a warning once and return an empty string so ingestion carries on.
"""
import logging
import re
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

AUDIO_EXT = {".mp3", ".wav", ".m4a", ".aac", ".ogg", ".oga", ".flac", ".wma", ".opus", ".amr"}
VIDEO_EXT = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".wmv", ".flv", ".3gp", ".mpeg", ".mpg", ".m4v"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".gif", ".webp"}


def media_kind(path_or_name) -> str:
    """Return 'audio', 'video', 'image', 'pdf' or 'other' from the file extension."""
    ext = Path(str(path_or_name)).suffix.lower()
    if ext in AUDIO_EXT:
        return "audio"
    if ext in VIDEO_EXT:
        return "video"
    if ext in IMAGE_EXT:
        return "image"
    if ext == ".pdf":
        return "pdf"
    return "other"


def clean_text(text: str) -> str:
    text = text.replace("\x0c", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------------------
# OCR (Tesseract, Tamil + English)
# ---------------------------------------------------------------------------
_ocr_state: dict = {"checked": False, "ok": False, "lang": config.OCR_LANG}


def _ocr_ready() -> bool:
    if _ocr_state["checked"]:
        return _ocr_state["ok"]
    _ocr_state["checked"] = True
    try:
        import pytesseract

        if config.TESSERACT_CMD:
            pytesseract.pytesseract.tesseract_cmd = config.TESSERACT_CMD
        installed = set(pytesseract.get_languages(config=""))
        wanted = [lang for lang in config.OCR_LANG.split("+") if lang]
        usable = [lang for lang in wanted if lang in installed]
        missing = [lang for lang in wanted if lang not in installed]
        if missing:
            log.warning(
                "Tesseract language pack(s) missing: %s. Install them for better OCR "
                "(Tamil = 'tam').", ", ".join(missing)
            )
        if not usable:
            usable = ["eng"] if "eng" in installed else []
        if not usable:
            raise RuntimeError("no usable Tesseract language packs")
        _ocr_state["lang"] = "+".join(usable)
        _ocr_state["ok"] = True
    except Exception as exc:  # pragma: no cover - depends on system install
        log.warning("Tesseract OCR not available (%s). Image OCR is disabled.", exc)
        _ocr_state["ok"] = False
    return _ocr_state["ok"]


def ocr_image(path: Path) -> str:
    """OCR an image file. Returns '' if OCR is disabled/unavailable or fails."""
    if not config.ENABLE_OCR or not _ocr_ready():
        return ""
    try:
        import pytesseract
        from PIL import Image

        with Image.open(path) as img:
            if img.mode not in ("RGB", "L"):
                img = img.convert("RGB")
            # Upscale small images; Tesseract works best at ~300 DPI equivalent.
            if min(img.size) < 800:
                scale = 800 / min(img.size)
                img = img.resize((int(img.width * scale), int(img.height * scale)))
            text = pytesseract.image_to_string(img, lang=_ocr_state["lang"])
        return clean_text(text)
    except Exception as exc:
        log.warning("OCR failed for %s: %s", Path(path).name, exc)
        return ""


# ---------------------------------------------------------------------------
# Speech to text (faster-whisper; reads audio and video files directly)
# ---------------------------------------------------------------------------
_whisper_state: dict = {"model": None, "failed": False}


def _whisper():
    if _whisper_state["model"] is not None or _whisper_state["failed"]:
        return _whisper_state["model"]
    try:
        from faster_whisper import WhisperModel

        log.info("Loading Whisper model '%s' (first run downloads it)...", config.WHISPER_MODEL)
        _whisper_state["model"] = WhisperModel(
            config.WHISPER_MODEL, device=config.WHISPER_DEVICE, compute_type="int8"
        )
    except Exception as exc:  # pragma: no cover - depends on install
        log.warning("faster-whisper not available (%s). Audio/video transcription disabled.", exc)
        _whisper_state["failed"] = True
    return _whisper_state["model"]


def transcribe(path: Path) -> str:
    """Transcribe an audio or video file (Tamil and English auto-detected)."""
    if not config.ENABLE_MEDIA:
        return ""
    model = _whisper()
    if model is None:
        return ""
    try:
        segments, info = model.transcribe(
            str(path),
            language=config.WHISPER_LANGUAGE or None,
            vad_filter=True,
            beam_size=5,
        )
        text = " ".join(seg.text.strip() for seg in segments if seg.text.strip())
        log.info(
            "Transcribed %s (%s, %.0fs)", Path(path).name, info.language, info.duration or 0
        )
        return clean_text(text)
    except Exception as exc:
        log.warning("Transcription failed for %s: %s", Path(path).name, exc)
        return ""
