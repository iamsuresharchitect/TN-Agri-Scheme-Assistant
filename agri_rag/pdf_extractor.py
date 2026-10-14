"""Extract text, tables, links, images and embedded files from a PDF (PyMuPDF)."""
import hashlib
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import fitz  # PyMuPDF

from . import config, media
from .models import Document

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s<>\"'()\[\]{}]+", re.IGNORECASE)
WWW_RE = re.compile(r"(?<![/\w])www\.[a-z0-9-]+(?:\.[a-z0-9-]+)+[^\s<>\"'()\[\]{}]*", re.IGNORECASE)


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def clean_url(url: str) -> str:
    url = url.strip().rstrip(".,;:!?'\"")
    url = url.split("#", 1)[0]
    if url.lower().startswith("www."):
        url = "https://" + url
    return url


def _safe_name(name: str) -> str:
    name = re.sub(r"[^\w.\-]+", "_", name, flags=re.UNICODE).strip("._")
    return name[:120] or "file"


@dataclass
class PDFExtraction:
    docs: list = field(default_factory=list)            # list[Document]
    links: dict = field(default_factory=dict)           # url -> first page it appears on
    attachments: list = field(default_factory=list)     # list[(Path, page|None)]
    stats: dict = field(default_factory=dict)


def extract_pdf(path: Path, source_label: str | None = None, parent_url: str | None = None) -> PDFExtraction:
    """Extract everything useful from one PDF.

    Text and tables become Documents. Images are OCR'd. Scanned pages are rendered
    and OCR'd. Links and embedded/attached files are returned for the pipeline to
    process (scrape / transcribe / recurse).
    """
    path = Path(path)
    label = source_label or path.name
    asset_dir = config.CACHE_DIR / "assets" / f"{_safe_name(path.stem)}_{file_hash(path)[:10]}"
    asset_dir.mkdir(parents=True, exist_ok=True)

    res = PDFExtraction()
    stats = {"pages": 0, "text_pages": 0, "ocr_pages": 0, "tables": 0, "images_ocr": 0,
             "links": 0, "attachments": 0}
    seen_images: set = set()

    with fitz.open(path) as pdf:
        if pdf.needs_pass:
            log.warning("Skipping password-protected PDF: %s", path.name)
            res.stats = stats
            return res

        for pno, page in enumerate(pdf, start=1):
            stats["pages"] += 1
            meta = {"source": label, "page": pno, "source_type": "pdf"}
            if parent_url:
                meta["url"] = parent_url
                meta["source_type"] = "web_pdf"

            raw_text = page.get_text("text") or ""
            text = media.clean_text(raw_text)
            page_was_ocrd = False

            # 1. Scanned page -> render at 300 DPI and OCR the whole page
            if len(text) < config.MIN_PAGE_TEXT and config.ENABLE_OCR:
                img_path = asset_dir / f"page_{pno:03d}.png"
                if not img_path.exists():
                    page.get_pixmap(dpi=300).save(img_path)
                ocr_text = media.ocr_image(img_path)
                if len(ocr_text) >= config.MIN_OCR_TEXT:
                    res.docs.append(Document(ocr_text, {**meta, "content_type": "page_ocr"}))
                    stats["ocr_pages"] += 1
                    page_was_ocrd = True

            # 2. Normal text layer
            if text and not page_was_ocrd:
                res.docs.append(Document(text, {**meta, "content_type": "text"}))
                stats["text_pages"] += 1

            # 3. Tables (kept as markdown so rows/columns stay together)
            try:
                for ti, table in enumerate(page.find_tables().tables, start=1):
                    md = table.to_markdown(clean=True).strip()
                    if md.count("|") > 4:
                        res.docs.append(Document(md, {**meta, "content_type": "table", "table": ti}))
                        stats["tables"] += 1
            except Exception as exc:
                log.debug("Table detection failed on %s p%s: %s", path.name, pno, exc)

            # 4. Links: clickable annotations + URLs typed in the text
            for link in page.get_links():
                uri = link.get("uri") or ""
                if uri.lower().startswith(("http://", "https://", "www.")):
                    res.links.setdefault(clean_url(uri), pno)
            for match in URL_RE.findall(raw_text) + WWW_RE.findall(raw_text):
                res.links.setdefault(clean_url(match), pno)

            # 5. Embedded images -> OCR (skip if the whole page was already OCR'd)
            if config.ENABLE_OCR and not page_was_ocrd:
                for img in page.get_images(full=True):
                    xref = img[0]
                    if xref in seen_images:
                        continue
                    seen_images.add(xref)
                    try:
                        info = pdf.extract_image(xref)
                    except Exception:
                        continue
                    if not info or min(info.get("width", 0), info.get("height", 0)) < config.MIN_IMAGE_PX:
                        continue
                    img_path = asset_dir / f"p{pno:03d}_img{xref}.{info.get('ext', 'png')}"
                    if not img_path.exists():
                        img_path.write_bytes(info["image"])
                    ocr_text = media.ocr_image(img_path)
                    if len(ocr_text) >= config.MIN_OCR_TEXT:
                        res.docs.append(Document(
                            ocr_text, {**meta, "content_type": "image_ocr", "image": img_path.name}
                        ))
                        stats["images_ocr"] += 1

            # 6. File-attachment annotations (audio/video/docs pinned to a page)
            for annot in page.annots() or []:
                try:
                    if annot.type[0] != fitz.PDF_ANNOT_FILE_ATTACHMENT:
                        continue
                    fname = annot.file_info.get("filename") or f"attachment_p{pno}"
                    out = asset_dir / _safe_name(fname)
                    out.write_bytes(annot.get_file())
                    res.attachments.append((out, pno))
                except Exception as exc:
                    log.debug("Could not read attachment on %s p%s: %s", path.name, pno, exc)

        # 7. Document-level embedded files
        for name in pdf.embfile_names():
            try:
                info = pdf.embfile_info(name)
                fname = info.get("filename") or name
                out = asset_dir / _safe_name(fname)
                out.write_bytes(pdf.embfile_get(name))
                res.attachments.append((out, None))
            except Exception as exc:
                log.debug("Could not read embedded file %s in %s: %s", name, path.name, exc)

    stats["links"] = len(res.links)
    stats["attachments"] = len(res.attachments)
    res.stats = stats
    log.info("Extracted %s: %s", label, stats)
    return res
