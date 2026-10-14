"""Streamlit UI for farmers: ask a question, get an answer grounded in the scheme documents.

Run:  streamlit run app.py
"""
import logging
import re
import uuid
from pathlib import Path

import streamlit as st

from agri_rag import config, tracing
from agri_rag.pipeline import IngestionPipeline
from agri_rag.rag import NOT_FOUND, RAGEngine, format_source
from agri_rag.vectorstore import Embedder, FaissStore

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

st.set_page_config(page_title=config.APP_TITLE, page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.5rem; max-width: 1100px;}
      .src-card {border-left: 3px solid #4c9a2a; padding: .4rem .8rem; margin: .4rem 0;
                 background: rgba(76,154,42,.07); border-radius: 4px; font-size: .9rem;}
      .src-meta {color: #5f6b5a; font-size: .8rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- cache
@st.cache_resource(show_spinner="Loading the knowledge base…")
def get_store() -> FaissStore:
    return FaissStore(Embedder()).load()


@st.cache_resource
def get_engine() -> RAGEngine:
    return RAGEngine(get_store())


try:
    store = get_store()
    engine = get_engine()
except Exception as exc:
    st.error(f"Could not load the knowledge base: {exc}")
    st.stop()

LANG_OPTIONS = {"Auto (match my question)": "auto", "தமிழ்": "ta", "English": "en"}
EXAMPLES = [
    "What subsidy is available for drip irrigation?",
    "Who is eligible for the PM-KISAN scheme?",
    "பயிர் காப்பீடு திட்டத்தில் சேர என்ன ஆவணங்கள் தேவை?",
    "How do I apply for a farm machinery subsidy?",
]

# --------------------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    lang_label = st.radio("Answer language / பதில் மொழி", list(LANG_OPTIONS))
    with st.expander("Retrieval settings"):
        top_k = st.slider("Chunks to retrieve", 2, 15, config.TOP_K)
        threshold = st.slider("Minimum similarity", 0.0, 1.0, config.SIM_THRESHOLD, 0.05,
                              help="Chunks scoring below this are ignored. If none pass, the app "
                                   "says the answer is not in the documents.")

    stats = store.stats()
    if tracing.ENABLED:
        st.caption(f"🔍 LangSmith tracing on · project `{config.LANGSMITH_PROJECT}`")
    st.divider()
    st.subheader("📚 Knowledge base")
    c1, c2 = st.columns(2)
    c1.metric("PDFs", stats["pdfs"])
    c2.metric("Chunks", stats["chunks"])
    if stats["by_type"]:
        with st.expander("Content breakdown"):
            for ct, n in sorted(stats["by_type"].items(), key=lambda x: -x[1]):
                st.write(f"{ct.replace('_', ' ')}: **{n}**")
            if store.manifest:
                st.caption("Ingested files:")
                for info in store.manifest.values():
                    st.caption(f"• {info['file']} ({info['chunks']} chunks)")

    # ---- admin: add documents
    st.divider()
    st.subheader("➕ Add scheme PDFs")
    unlocked = True
    if config.ADMIN_PASSWORD:
        pw = st.text_input("Admin password", type="password")
        unlocked = pw == config.ADMIN_PASSWORD
        if pw and not unlocked:
            st.error("Wrong password")
    if unlocked:
        uploads = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
        opt_urls = st.checkbox("Scrape links inside PDFs", value=config.ENABLE_URL_SCRAPING)
        opt_ocr = st.checkbox("OCR images & scanned pages", value=config.ENABLE_OCR)
        opt_media = st.checkbox("Transcribe audio / video", value=config.ENABLE_MEDIA)
        col_a, col_b = st.columns(2)
        add_clicked = col_a.button("Add to knowledge base", type="primary", disabled=not uploads)
        rescan_clicked = col_b.button("Re-scan data/pdfs",
                                      help="Ingest any new PDFs copied into the data/pdfs folder")

        if add_clicked or rescan_clicked:
            config.ENABLE_URL_SCRAPING, config.ENABLE_OCR, config.ENABLE_MEDIA = opt_urls, opt_ocr, opt_media
            paths: list = []
            if add_clicked:
                for up in uploads:
                    dest = config.PDF_DIR / Path(up.name).name
                    dest.write_bytes(up.getbuffer())
                    paths.append(dest)
            else:
                paths = sorted(config.PDF_DIR.glob("**/*.pdf"))
            with st.status("Processing documents…", expanded=True) as status:
                report = IngestionPipeline(store, progress=st.write).run(paths)
                label = (f"Added {report['pdfs_processed']} PDF(s), {report['chunks_added']} chunks, "
                         f"{report['urls_scraped']} links scraped")
                status.update(label=label, state="error" if report["errors"] else "complete")
                for err in report["errors"]:
                    st.error(err)
            st.rerun()

        st.markdown("**🌐 Add web pages**")
        url_text = st.text_area("One URL per line (web page, online PDF or YouTube)",
                                placeholder="https://fert.gov.in/en", height=90)
        follow = st.checkbox("Also load same-site pages it links to", value=True,
                             help=f"Loads up to {config.MAX_LINKS_PER_PDF} linked pages/PDFs on the same website.")
        if st.button("Add URLs to knowledge base", disabled=not url_text.strip()):
            urls = [u.strip() for u in url_text.splitlines() if u.strip()]
            with st.status("Fetching web pages…", expanded=True) as status:
                report = IngestionPipeline(store, progress=st.write).run_urls(urls, follow_links=follow)
                status.update(label=(f"Added {report['urls_processed']} URL(s), {report['chunks_added']} chunks "
                                     f"from {report['urls_scraped']} pages"),
                              state="error" if report["errors"] else "complete")
                for err in report["errors"]:
                    st.error(err)
            if not report["errors"]:
                st.rerun()

    st.divider()
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------------------------------- main
st.title("🌾 " + config.APP_TITLE)
st.caption("Ask about Tamil Nadu agriculture schemes in Tamil or English. Answers come only from the "
           "official scheme documents loaded into this app. · தமிழ்நாடு வேளாண் திட்டங்கள் பற்றி "
           "தமிழிலோ ஆங்கிலத்திலோ கேளுங்கள்.")

if stats["chunks"] == 0:
    st.info("The knowledge base is empty. Upload scheme PDFs from the sidebar, or copy them into "
            "`data/pdfs` and run `python ingest.py`.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_sources(hits: list):
    if not hits:
        return
    with st.expander(f"📄 Sources ({len(hits)})"):
        for i, (score, chunk) in enumerate(hits, start=1):
            meta = chunk["metadata"]
            snippet = chunk["text"][:350].replace("\n", " ")
            link = f' · <a href="{meta["url"]}" target="_blank">open link</a>' if meta.get("url") else ""
            found = f' · found in {meta["found_in"]}' if meta.get("found_in") else ""
            st.markdown(
                f'<div class="src-card"><b>[{i}] {format_source(meta)}</b>'
                f'<div class="src-meta">similarity {score:.2f}{found}{link}</div>'
                f"{snippet}…</div>",
                unsafe_allow_html=True,
            )


def render_near_misses(near: list, threshold: float):
    """When nothing passed the threshold, show what came closest so the threshold can be tuned."""
    if not near:
        return
    with st.expander(f"🔎 Why no answer? Closest matches (best {near[0][0]:.2f}, threshold {threshold:.2f})"):
        if near[0][0] < threshold:
            st.caption("None of these scored above the minimum similarity, so Claude was not asked. "
                       "If they look relevant, lower **Minimum similarity** in the sidebar and ask again.")
        else:
            st.caption("Claude read these but they don't answer the question. The information is "
                       "probably not in the loaded documents.")
        for i, (score, chunk) in enumerate(near, start=1):
            snippet = chunk["text"][:300].replace("\n", " ")
            st.markdown(
                f'<div class="src-card"><b>{format_source(chunk["metadata"])}</b>'
                f'<div class="src-meta">similarity {score:.2f}</div>{snippet}…</div>',
                unsafe_allow_html=True,
            )


def _on_feedback(run_id: str, key: str):
    value = st.session_state.get(key)
    if value is not None and tracing.send_feedback(run_id, score=value):
        st.toast("Thanks for the feedback!")


def render_feedback(msg: dict):
    """👍/👎 under an answer; the rating is attached to the question's LangSmith trace."""
    if tracing.ENABLED and msg.get("run_id"):
        key = f"fb_{msg['run_id']}"
        st.feedback("thumbs", key=key, on_change=_on_feedback, args=(msg["run_id"], key))


# example questions on an empty chat
pending = None
if not st.session_state.messages and stats["chunks"] > 0:
    st.write("**Try asking:**")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex{i}", use_container_width=True):
            pending = q

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍🌾" if msg["role"] == "user" else "🌾"):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources", []))
            render_near_misses(msg.get("near", []), msg.get("threshold", threshold))
            render_feedback(msg)

question = st.chat_input("Type your question… / உங்கள் கேள்வியை எழுதுங்கள்…") or pending

if question:
    history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages]
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑‍🌾"):
        st.markdown(question)

    run_id = str(uuid.uuid4())   # LangSmith trace id for this question
    is_url = re.fullmatch(r"\s*(https?://|www\.)\S+\s*", question) is not None
    with st.chat_message("assistant", avatar="🌾"):
        if is_url:
            answer, hits, near = ("That looks like a web link, not a question. To load that page into the "
                                  "knowledge base, paste it under **➕ Add scheme PDFs → 🌐 Add web pages** "
                                  "in the sidebar, then ask your question here."), [], []
            st.markdown(answer)
        elif stats["chunks"] == 0:
            answer, hits, near = "Please add scheme PDFs to the knowledge base first.", [], []
            st.markdown(answer)
        else:
            near: list = []
            with st.spinner("Searching the scheme documents…"):
                stream, hits, _ = engine.answer_stream(
                    question, language=LANG_OPTIONS[lang_label], k=top_k,
                    threshold=threshold, history=history, run_id=run_id, near_misses=near,
                )
            answer = st.write_stream(stream)
            # Claude read the chunks but said the answer isn't in them: show them as
            # "closest matches", not as sources of an answer.
            if hits and any(answer.strip().startswith(t[:40]) for t in NOT_FOUND.values()):
                near, hits = hits[:3], []
            render_sources(hits)
            render_near_misses(near, threshold)
            render_feedback({"run_id": run_id})
    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": hits,
                                      "near": near, "threshold": threshold, "run_id": run_id})
