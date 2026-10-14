"""Ingestion pipeline: PDFs -> (text, tables, OCR, links, media) -> chunks -> FAISS."""
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable

from . import config, media
from .chunker import chunk_documents
from .models import Document
from .pdf_extractor import extract_pdf, file_hash
from .url_scraper import Scraper
from .vectorstore import FaissStore

log = logging.getLogger(__name__)


class IngestionPipeline:
    def __init__(self, store: FaissStore, progress: Callable[[str], None] | None = None):
        self.store = store
        self.progress = progress or (lambda msg: log.info(msg))
        self.scraper: Scraper | None = None
        self.report: dict = {}

    # ------------------------------------------------------------------ public
    def run(self, pdf_paths: list, force: bool = False) -> dict:
        pdf_paths = [Path(p) for p in pdf_paths if Path(p).suffix.lower() == ".pdf"]
        self.report = {"pdfs_processed": 0, "pdfs_skipped": 0, "chunks_added": 0,
                       "urls_scraped": 0, "linked_pdfs": 0, "media_transcribed": 0,
                       "by_type": {}, "errors": []}
        self.scraper = Scraper() if config.ENABLE_URL_SCRAPING else None
        try:
            for n, path in enumerate(pdf_paths, start=1):
                sha = file_hash(path)
                if sha in self.store.manifest and not force:
                    self.report["pdfs_skipped"] += 1
                    self.progress(f"[{n}/{len(pdf_paths)}] {path.name}: already in knowledge base, skipped")
                    continue
                self.progress(f"[{n}/{len(pdf_paths)}] Reading {path.name}")
                try:
                    docs = self._process_pdf(path, label=path.name, depth=0)
                    chunks = chunk_documents(docs)
                    self.progress(f"[{n}/{len(pdf_paths)}] Embedding {len(chunks)} chunks from {path.name}")
                    added = self.store.add(chunks)
                    for d in docs:
                        ct = d.metadata.get("content_type", "text")
                        self.report["by_type"][ct] = self.report["by_type"].get(ct, 0) + 1
                    self.store.manifest[sha] = {"file": path.name, "chunks": added,
                                                "ingested_at": datetime.now().isoformat(timespec="seconds")}
                    self.store.save()   # save after every PDF so a crash loses little
                    self.report["pdfs_processed"] += 1
                    self.report["chunks_added"] += added
                except Exception as exc:
                    log.exception("Failed on %s", path.name)
                    self.report["errors"].append(f"{path.name}: {exc}")
        finally:
            if self.scraper:
                self.scraper.close()
                self._save_failures()
        self.progress(
            f"Done: {self.report['pdfs_processed']} PDF(s) added, {self.report['chunks_added']} chunks, "
            f"{self.report['urls_scraped']} links scraped"
        )
        return self.report

    def run_urls(self, urls: list, force: bool = False, follow_links: bool = True) -> dict:
        """Load web pages / online PDFs / media / YouTube links straight into the knowledge base.

        follow_links: for a web page, also load up to MAX_LINKS_PER_PDF pages and PDFs on the
                      same website that it links to (one level deep).
        """
        urls = [u.strip() for u in urls if u and u.strip()]
        self.report = {"urls_processed": 0, "urls_skipped": 0, "chunks_added": 0,
                       "urls_scraped": 0, "linked_pdfs": 0, "media_transcribed": 0,
                       "by_type": {}, "errors": []}
        self.scraper = Scraper()   # always on here: the user asked for these URLs explicitly
        try:
            for n, url in enumerate(urls, start=1):
                if not url.lower().startswith(("http://", "https://")):
                    url = "https://" + url
                key = "url:" + url
                if key in self.store.manifest and not force:
                    self.report["urls_skipped"] += 1
                    self.progress(f"[{n}/{len(urls)}] {url}: already in knowledge base, skipped")
                    continue
                self.progress(f"[{n}/{len(urls)}] Fetching {url}")
                try:
                    docs = self._process_url(url, follow_links)
                    if not docs:
                        raise RuntimeError("no readable content (site unreachable, blocked by robots.txt, "
                                           "or the page has no text)")
                    chunks = chunk_documents(docs)
                    self.progress(f"[{n}/{len(urls)}] Embedding {len(chunks)} chunks from {url}")
                    added = self.store.add(chunks)
                    for d in docs:
                        ct = d.metadata.get("content_type", "text")
                        self.report["by_type"][ct] = self.report["by_type"].get(ct, 0) + 1
                    self.store.manifest[key] = {"file": url, "chunks": added,
                                                "ingested_at": datetime.now().isoformat(timespec="seconds")}
                    self.store.save()
                    self.report["urls_processed"] += 1
                    self.report["chunks_added"] += added
                except Exception as exc:
                    log.exception("Failed on %s", url)
                    self.report["errors"].append(f"{url}: {exc}")
        finally:
            self.scraper.close()
            self._save_failures()
        self.progress(f"Done: {self.report['urls_processed']} URL(s) added, {self.report['chunks_added']} chunks, "
                      f"{self.report['urls_scraped']} pages fetched")
        return self.report

    # ----------------------------------------------------------------- helpers
    def _save_failures(self):
        """Put links that couldn't be fetched in the report and in data/failed_links.csv,
        so they can be downloaded by hand and dropped into data/pdfs."""
        import csv

        failures = list(self.scraper.failures) if self.scraper else []
        self.report["links_failed"] = len(failures)
        self.report["failed_links"] = [{"url": u, "reason": r} for u, r in failures]
        if failures:
            path = config.DATA_DIR / "failed_links.csv"
            with open(path, "w", newline="", encoding="utf-8-sig") as fh:
                w = csv.writer(fh)
                w.writerow(["url", "reason"])
                w.writerows(failures)
            self.report["failed_links_file"] = str(path)

    def _fetch_into(self, url: str, docs: list, depth: int, found_in: str | None = None):
        """Fetch one URL and append its documents (and any linked PDF's content) to docs."""
        result = self.scraper.fetch(url)
        if result is None:
            return None
        self.report["urls_scraped"] += 1
        for d in result.docs:
            if found_in:
                d.metadata["found_in"] = found_in
            if d.metadata.get("content_type", "").endswith("_transcript"):
                self.report["media_transcribed"] += 1
        docs.extend(result.docs)
        if result.pdf_path:
            self.report["linked_pdfs"] += 1
            self.progress(f"  PDF: {url}")
            child = self._process_pdf(result.pdf_path, label=url, depth=depth + 1, parent_url=url)
            for d in child:
                if found_in:
                    d.metadata.setdefault("found_in", found_in)
            docs.extend(child)
        return result

    def _process_url(self, url: str, follow_links: bool) -> list:
        docs: list = []
        result = self._fetch_into(url, docs, depth=0)
        if result and follow_links and result.links:
            links = result.links[: config.MAX_LINKS_PER_PDF]
            self.progress(f"  following {len(links)} link(s) on the same site")
            for link in links:
                self._fetch_into(link, docs, depth=1, found_in=url)
        return docs

    def _process_pdf(self, path: Path, label: str, depth: int, parent_url: str | None = None) -> list:
        ex = extract_pdf(path, source_label=label, parent_url=parent_url)
        docs = list(ex.docs)

        for att_path, page in ex.attachments:
            docs.extend(self._process_file(att_path, label, page, depth))

        if self.scraper and depth < config.CRAWL_DEPTH and ex.links:
            links = list(ex.links.items())[: config.MAX_LINKS_PER_PDF]
            self.progress(f"  {label}: scraping {len(links)} link(s)")
            for url, page in links:
                result = self.scraper.fetch(url)
                if result is None:
                    continue
                self.report["urls_scraped"] += 1
                found_in = f"{label} (page {page})"
                for d in result.docs:
                    d.metadata["found_in"] = found_in
                    if d.metadata.get("content_type", "").endswith("_transcript"):
                        self.report["media_transcribed"] += 1
                docs.extend(result.docs)
                if result.pdf_path:
                    self.report["linked_pdfs"] += 1
                    self.progress(f"  Linked PDF: {url}")
                    child = self._process_pdf(result.pdf_path, label=url, depth=depth + 1, parent_url=url)
                    for d in child:
                        d.metadata.setdefault("found_in", found_in)
                    docs.extend(child)
        return docs

    def _process_file(self, path: Path, parent: str, page, depth: int) -> list:
        """Handle a file embedded in / attached to a PDF."""
        kind = media.media_kind(path)
        meta = {"source": f"{parent} → {path.name}", "source_type": "attachment", "found_in": parent}
        if page:
            meta["page"] = page
        if kind in ("audio", "video"):
            self.progress(f"  Transcribing {kind}: {path.name}")
            text = media.transcribe(path)
            if text:
                self.report["media_transcribed"] += 1
                return [Document(text, {**meta, "content_type": f"{kind}_transcript"})]
        elif kind == "image":
            text = media.ocr_image(path)
            if len(text) >= config.MIN_OCR_TEXT:
                return [Document(text, {**meta, "content_type": "image_ocr"})]
        elif kind == "pdf":
            return self._process_pdf(path, label=meta["source"], depth=depth)
        elif path.suffix.lower() in (".txt", ".csv", ".md"):
            text = path.read_text(encoding="utf-8", errors="ignore").strip()
            if text:
                return [Document(text, {**meta, "content_type": "text"})]
        else:
            log.info("Unsupported attachment type skipped: %s", path.name)
        return []
