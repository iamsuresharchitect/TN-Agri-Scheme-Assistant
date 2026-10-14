# 🌾 TN Agri Scheme Assistant

A RAG (retrieval-augmented generation) app that answers Tamil Nadu farmers' questions about
agriculture schemes, in Tamil or English, **using only the scheme documents you load**.

```
PDFs ──► text · tables · scanned-page OCR · image OCR · embedded audio/video
  │
  └─► links inside PDFs ──► web pages (requests / Playwright)
                         ├─► linked PDFs (fully extracted too)
                         ├─► audio / video files (Whisper transcription)
                         └─► YouTube videos (yt-dlp + Whisper)
                                      │
                         chunk ─► embed (BAAI/bge-m3) ─► FAISS
                                      │
Farmer's question ─► embed ─► FAISS top-k ─► similarity check ─► Claude (grounded prompt) ─► Streamlit
```

## How "answers only from the vector DB" is enforced

1. **No web search.** Claude is called with no tools. It only sees the retrieved chunks.
2. **Similarity gate.** If no chunk scores above `SIM_THRESHOLD`, the app returns a fixed
   "not found in the documents" message and **does not call Claude at all**.
3. **Strict prompt.** Claude is told to use only the `<context>` text, never invent amounts,
   dates or eligibility rules, and to reply with the "not found" sentence when the answer is missing.
4. **Citations.** Every fact carries a `[n]` reference, and the UI shows each source
   (PDF name and page, or URL) so answers can be checked.

## Project layout

```
tn-agri-rag/
├── app.py                  Streamlit UI (chat + upload panel)
├── ingest.py               CLI: load PDFs into FAISS
├── ask.py                  CLI: ask a question (testing)
├── requirements.txt
├── .env.example
├── agri_rag/
│   ├── config.py           all settings (read from .env)
│   ├── pdf_extractor.py    text, tables, links, images, attachments (PyMuPDF)
│   ├── url_scraper.py      web pages, linked PDFs, media, YouTube
│   ├── media.py            Tesseract OCR + faster-whisper transcription
│   ├── chunker.py          paragraph/sentence-aware chunking
│   ├── vectorstore.py      FAISS store + multilingual embedder
│   ├── pipeline.py         ingestion orchestration
│   ├── rag.py              retrieval + Claude answer generation
│   └── tracing.py          optional LangSmith tracing + feedback
└── data/
    ├── pdfs/               put your scheme PDFs here
    ├── cache/              extracted images, downloads (auto)
    └── index/              index.faiss, chunks.jsonl, manifest.json (auto)
```

## Setup

### 1. Python packages (Python 3.10+)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

### 2. Tesseract OCR with Tamil

- **Windows:** install from the UB Mannheim build (github.com/UB-Mannheim/tesseract/wiki).
  During setup, tick **Tamil** under *Additional language data*. If `tesseract` isn't on PATH,
  set `TESSERACT_CMD` in `.env`.
- **Ubuntu/Debian:** `sudo apt install tesseract-ocr tesseract-ocr-tam`

Check it with `tesseract --list-langs`. You should see `tam` and `eng`.

### 3. ffmpeg (only for YouTube links)

yt-dlp uses ffmpeg for some formats. Windows: `winget install ffmpeg`. Ubuntu: `sudo apt install ffmpeg`.

### 4. API key

```bash
copy .env.example .env      # Windows   (Linux: cp .env.example .env)
```
Edit `.env` and set `ANTHROPIC_API_KEY`.

> **First run downloads models:** `BAAI/bge-m3` (~2.3 GB) and Whisper `small` (~480 MB).
> On a machine without a GPU, set `WHISPER_MODEL=base` for faster transcription.

## Usage

### Load documents
```bash
# copy scheme PDFs into data/pdfs, then:
python ingest.py

python ingest.py --no-urls --no-media   # quick first test
python ingest.py --rebuild               # wipe and re-ingest everything
```
Already-ingested PDFs are recognised by content hash and skipped on later runs.

### Load web pages directly
```bash
python ingest.py --urls https://fert.gov.in/en
python ingest.py --urls https://fert.gov.in/en --no-follow   # only that page
```
Or paste URLs (one per line) under **🌐 Add web pages** in the sidebar. The page is loaded along
with up to `MAX_LINKS_PER_PDF` pages and PDFs on the same website that it links to. Online PDFs,
audio/video files and YouTube links work too. Typing a URL into the chat box doesn't load it;
the chat only answers questions.

Sites that block automated access in `robots.txt` are skipped. Set `RESPECT_ROBOTS=false`
in `.env` only if you have permission to scrape that site.

### Run the app
```bash
streamlit run app.py
```
Open http://localhost:8501. You can also upload PDFs from the sidebar. Set `ADMIN_PASSWORD`
in `.env` before sharing the app so farmers can't change the knowledge base.

### Test from the command line
```bash
python ask.py "What subsidy is available for drip irrigation?" --show-context
python ask.py "சொட்டு நீர் பாசனத்திற்கு மானியம் உண்டா?"
```

## LangSmith tracing (optional)

Every farmer question can be recorded in LangSmith, so you can see exactly what was
retrieved and what Claude answered. No LangChain is needed.

```
TN Agri RAG (chain)                  question, language, answer
  ├── FAISS retrieval (retriever)    each chunk, its source and similarity score
  └── Claude (llm)                   full prompt, answer, tokens, latency
```

1. Create an API key at smith.langchain.com (Settings → API Keys).
2. In `.env`, set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY=...`.
3. Restart the app. The sidebar shows "LangSmith tracing on".

When tracing is on, each answer gets 👍/👎 buttons. Ratings are attached to the trace as
`farmer_rating` feedback, so you can filter LangSmith for answers farmers disliked. Questions
answered with "not found" appear too (with no Claude call), which shows gaps in your documents.

Tracing sends questions, retrieved document text and answers to LangSmith's cloud. Leave it
off if that isn't acceptable for your data.

## When the app says "not found"

Open **🔎 Why no answer?** under the reply. It lists the closest chunks and their scores.
For more detail, run:

```bash
python ask.py "Distribution of Certified Seeds of maize" --debug
```

This lists what's in the knowledge base, the top 10 matches with scores (no Claude call), and
whether any chunk contains all the words of your question.

- **Matches look right but score below the threshold:** lower `SIM_THRESHOLD`.
- **No chunk contains the words:** the PDF isn't ingested, or its pages are scanned images and
  OCR isn't working. Check that the file is listed and that `tesseract --list-langs` works.

## Tuning

| Symptom | Try |
|---|---|
| Too many "not found" answers | Lower `SIM_THRESHOLD` (e.g. 0.35) or raise `TOP_K` |
| Answers mix in unrelated schemes | Raise `SIM_THRESHOLD` (e.g. 0.55) |
| Tables answered badly | Raise `CHUNK_SIZE` so tables stay whole |
| Ingestion slow | `--no-media`, smaller `WHISPER_MODEL`, `EMBED_DEVICE=cuda` |
| Scanned Tamil PDFs read poorly | Confirm the `tam` Tesseract pack is installed |

If you change `EMBED_MODEL`, run `python ingest.py --rebuild`, because vectors from
different models can't be mixed.

## Notes on media in PDFs

Most scheme PDFs don't embed audio or video directly. Media usually appears as **links**
(YouTube, or .mp4/.mp3 files on a government site), which are scraped and transcribed. True
embedded files and file-attachment annotations are also extracted and transcribed.
Rich-media (Flash/3D) annotations are not supported.

## When links fail to load

Government websites often fail in ways the app works around automatically:

| Problem on the website | What the app does |
|---|---|
| Missing intermediate certificate ("unable to get local issuer certificate") | Checks certificates with the Windows certificate store (`truststore`) |
| Expired or mismatched certificate, old TLS | Retries that public page without verification (`SSL_FALLBACK=true`) |
| 403 "Forbidden" for non-browser clients | Retries with the headless browser |
| JavaScript-only pages | Renders them with the headless browser |
| Link broken across two lines in the PDF | Skips it |

The headless browser needs a one-time setup: `playwright install chromium`.

After each run, links that still failed are listed by reason, with the full list saved to
`data/failed_links.csv`. Download any important ones by hand into `data/pdfs`.

## Responsible scraping

Link scraping respects `robots.txt`, waits `REQUEST_DELAY` seconds between hits to the same
site, follows only links found in your PDFs (`CRAWL_DEPTH=1`), and skips social-media links.
