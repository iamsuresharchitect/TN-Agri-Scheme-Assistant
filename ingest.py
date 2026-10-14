"""Command-line ingestion.

Examples:
    python ingest.py                       # ingest every PDF in data/pdfs (new ones only)
    python ingest.py --files a.pdf b.pdf   # ingest specific files
    python ingest.py --rebuild             # wipe the index and ingest everything again
    python ingest.py --no-urls --no-media  # quick run: skip link scraping and transcription
    python ingest.py --urls https://fert.gov.in/en https://example.gov.in/scheme.pdf
                                           # load web pages / online PDFs directly
    python ingest.py --urls https://fert.gov.in/en --no-follow   # only that page, not its links
"""
import argparse
import json
import logging
from pathlib import Path

from agri_rag import config
from agri_rag.pipeline import IngestionPipeline
from agri_rag.vectorstore import Embedder, FaissStore


def main():
    ap = argparse.ArgumentParser(description="Load TN agriculture scheme PDFs into the FAISS vector store")
    ap.add_argument("--pdf-dir", type=Path, default=config.PDF_DIR, help="folder with PDFs (default data/pdfs)")
    ap.add_argument("--files", nargs="*", type=Path, help="specific PDF files instead of a folder")
    ap.add_argument("--urls", nargs="+", help="web pages / online PDFs / YouTube links to load directly")
    ap.add_argument("--no-follow", action="store_true",
                    help="with --urls: load only the given pages, not the same-site pages they link to")
    ap.add_argument("--force", action="store_true", help="re-process PDFs already in the index")
    ap.add_argument("--rebuild", action="store_true", help="delete the index and start fresh")
    ap.add_argument("--no-urls", action="store_true", help="do not scrape links found in PDFs")
    ap.add_argument("--no-ocr", action="store_true", help="skip OCR of images / scanned pages")
    ap.add_argument("--no-media", action="store_true", help="skip audio/video transcription")
    ap.add_argument("--no-playwright", action="store_true", help="do not use a headless browser")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")

    config.ENABLE_URL_SCRAPING &= not args.no_urls
    config.ENABLE_OCR &= not args.no_ocr
    config.ENABLE_MEDIA &= not args.no_media
    config.ENABLE_PLAYWRIGHT &= not args.no_playwright

    store = FaissStore(Embedder())
    if args.rebuild and not args.urls:
        store.reset()
    else:
        store.load()

    if args.urls:
        report = IngestionPipeline(store).run_urls(args.urls, force=args.force,
                                                   follow_links=not args.no_follow)
    else:
        pdfs = args.files or sorted(args.pdf_dir.glob("**/*.pdf"))
        if not pdfs:
            print(f"No PDFs found in {args.pdf_dir}. Copy your scheme PDFs there and run again.")
            return
        report = IngestionPipeline(store).run(pdfs, force=args.force or args.rebuild)
    failed = report.pop("failed_links", [])
    print("\n=== Ingestion report ===")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if failed:
        from collections import Counter

        print(f"\n{len(failed)} link(s) could not be fetched:")
        for reason, n in Counter(f["reason"] for f in failed).most_common():
            print(f"  {n:4}  {reason}")
        print(f"Full list: {report.get('failed_links_file')}  (open in Excel; download any important "
              "ones by hand into data/pdfs)")
    print("\nKnowledge base:", json.dumps(store.stats(), ensure_ascii=False))


if __name__ == "__main__":
    main()
