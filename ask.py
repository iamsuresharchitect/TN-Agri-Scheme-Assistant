"""Ask a question from the command line (handy for testing retrieval).

    python ask.py "What subsidy is available for drip irrigation?"
    python ask.py "சொட்டு நீர் பாசனத்திற்கு மானியம் உண்டா?" --show-context
    python ask.py "Distribution of Certified Seeds of maize" --debug     # no Claude call

--debug lists the closest chunks and their similarity scores WITHOUT calling Claude,
whatever the threshold, plus what is in the knowledge base. Use it when the app says
"not found" and you expected an answer.
"""
import argparse
import logging

from agri_rag import config
from agri_rag.rag import RAGEngine, format_source
from agri_rag.vectorstore import Embedder, FaissStore


def debug(store: FaissStore, question: str, k: int):
    stats = store.stats()
    print(f"Knowledge base: {stats['chunks']} chunks from {stats['pdfs']} PDF(s)")
    for info in store.manifest.values():
        print(f"  - {info['file']}: {info['chunks']} chunks")
    print(f"\nTop {k} matches for: {question!r}   (threshold SIM_THRESHOLD={config.SIM_THRESHOLD})\n")
    for i, (score, chunk) in enumerate(store.search(question, k), 1):
        mark = "PASS" if score >= config.SIM_THRESHOLD else "below"
        print(f"[{i}] {score:.3f} {mark:5}  {format_source(chunk['metadata'])}")
        print("      " + chunk["text"][:250].replace("\n", " ") + "...\n")

    words = [w for w in question.lower().split() if len(w) > 3]
    if words:
        hits = [c for c in store.chunks if all(w in c["text"].lower() for w in words)]
        print(f"Keyword check: {len(hits)} chunk(s) contain all of {words}")
        for c in hits[:5]:
            print(f"  - {format_source(c['metadata'])}")
        if not hits:
            print("  -> The text may not be in the index (not ingested, scanned page without OCR,\n"
                  "     or worded differently in the PDF).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--lang", choices=["auto", "ta", "en"], default="auto")
    ap.add_argument("--show-context", action="store_true", help="print retrieved chunks")
    ap.add_argument("--debug", action="store_true", help="show closest matches + scores, no Claude call")
    ap.add_argument("-k", type=int, default=10, help="matches to show with --debug")
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    store = FaissStore(Embedder()).load()
    if args.debug:
        debug(store, args.question, args.k)
        return

    engine = RAGEngine(store)
    near: list = []
    stream, hits, _ = engine.answer_stream(args.question, language=args.lang, near_misses=near)
    for piece in stream:
        print(piece, end="", flush=True)
    print("\n\nSources:")
    for i, (score, chunk) in enumerate(hits, 1):
        print(f"  [{i}] {format_source(chunk['metadata'])}  (similarity {score:.2f})")
        if args.show_context:
            print("      " + chunk["text"][:400].replace("\n", " ") + "...")
    if near:
        print(f"  (none above threshold {config.SIM_THRESHOLD}; closest scored {near[0][0]:.2f} - "
              "try --debug)")


if __name__ == "__main__":
    main()
