"""FAISS vector store with a multilingual sentence-transformers embedder.

Files written to data/index/:
    index.faiss     the FAISS index (inner product on normalised vectors = cosine)
    chunks.jsonl    one line per vector: {"text": ..., "metadata": {...}}
    manifest.json   which PDFs have already been ingested (by SHA-256)
"""
import hashlib
import json
import logging
import re
from pathlib import Path

import faiss
import numpy as np

from . import config
from .models import Document

log = logging.getLogger(__name__)


class Embedder:
    """Wraps a sentence-transformers model. Default BAAI/bge-m3 handles Tamil + English."""

    def __init__(self, model_name: str = None, device: str = None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.EMBED_MODEL
        log.info("Loading embedding model %s (first run downloads it)...", self.model_name)
        self.model = SentenceTransformer(self.model_name, device=device or config.EMBED_DEVICE)
        self.dim = self.model.get_sentence_embedding_dimension()
        self._e5 = "e5" in self.model_name.lower()

    def _encode(self, texts: list) -> np.ndarray:
        vecs = self.model.encode(texts, batch_size=config.EMBED_BATCH_SIZE, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False)
        return np.asarray(vecs, dtype="float32")

    def embed_documents(self, texts: list) -> np.ndarray:
        if self._e5:
            texts = ["passage: " + t for t in texts]
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        if self._e5:
            text = "query: " + text
        return self._encode([text])


def _text_key(text: str) -> str:
    norm = re.sub(r"\s+", " ", text).strip().lower()
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()


def _embedding_text(doc: Document) -> str:
    """Prefix the chunk with its source name so retrieval can use the scheme/document title."""
    src = str(doc.metadata.get("source", ""))
    if src and not src.startswith("http"):
        return f"{Path(src).stem.replace('_', ' ')}\n{doc.text}"
    return doc.text


class FaissStore:
    def __init__(self, embedder: Embedder, index_dir: Path = None):
        self.embedder = embedder
        self.index_dir = Path(index_dir or config.INDEX_DIR)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.index = None
        self.chunks: list = []      # parallel to FAISS ids
        self.manifest: dict = {}    # file sha256 -> info
        self._keys: set = set()

    def __repr__(self):
        return f"FaissStore(chunks={len(self.chunks)})"

    # ------------------------------------------------------------ paths
    @property
    def _index_path(self):
        return self.index_dir / "index.faiss"

    @property
    def _chunks_path(self):
        return self.index_dir / "chunks.jsonl"

    @property
    def _manifest_path(self):
        return self.index_dir / "manifest.json"

    # ------------------------------------------------------- persistence
    def load(self) -> "FaissStore":
        if self._index_path.exists() and self._chunks_path.exists():
            self.index = faiss.read_index(str(self._index_path))
            if self.index.d != self.embedder.dim:
                raise RuntimeError(
                    f"Index was built with a {self.index.d}-dim model but EMBED_MODEL "
                    f"'{self.embedder.model_name}' is {self.embedder.dim}-dim. "
                    "Rebuild with: python ingest.py --rebuild"
                )
            with open(self._chunks_path, encoding="utf-8") as fh:
                self.chunks = [json.loads(line) for line in fh if line.strip()]
            if len(self.chunks) != self.index.ntotal:
                raise RuntimeError("chunks.jsonl and index.faiss are out of sync - run: python ingest.py --rebuild")
            self._keys = {_text_key(c["text"]) for c in self.chunks}
        if self._manifest_path.exists():
            self.manifest = json.loads(self._manifest_path.read_text(encoding="utf-8"))
        log.info("Vector store loaded: %d chunks from %d PDFs", len(self.chunks), len(self.manifest))
        return self

    def save(self):
        if self.index is not None:
            tmp = self._index_path.with_suffix(".tmp")
            faiss.write_index(self.index, str(tmp))
            tmp.replace(self._index_path)
        tmp = self._chunks_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            for c in self.chunks:
                fh.write(json.dumps(c, ensure_ascii=False) + "\n")
        tmp.replace(self._chunks_path)
        self._manifest_path.write_text(json.dumps(self.manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def reset(self):
        self.index, self.chunks, self.manifest, self._keys = None, [], {}, set()
        for p in (self._index_path, self._chunks_path, self._manifest_path):
            p.unlink(missing_ok=True)

    # ------------------------------------------------------------ write
    def add(self, docs: list) -> int:
        """Embed and add chunks, skipping exact duplicates. Returns how many were added."""
        new = []
        for d in docs:
            key = _text_key(d.text)
            if key not in self._keys:
                self._keys.add(key)
                new.append(d)
        if not new:
            return 0
        vecs = self.embedder.embed_documents([_embedding_text(d) for d in new])
        if self.index is None:
            self.index = faiss.IndexFlatIP(vecs.shape[1])
        self.index.add(vecs)
        self.chunks.extend({"text": d.text, "metadata": d.metadata} for d in new)
        return len(new)

    # ------------------------------------------------------------- read
    def search(self, query: str, k: int = None) -> list:
        """Return [(score, chunk_dict), ...] best first. Score is cosine similarity (0-1)."""
        if self.index is None or self.index.ntotal == 0:
            return []
        k = min(k or config.TOP_K, self.index.ntotal)
        qvec = self.embedder.embed_query(query)
        scores, ids = self.index.search(qvec, k)
        return [(float(s), self.chunks[i]) for s, i in zip(scores[0], ids[0]) if i != -1]

    def stats(self) -> dict:
        by_type: dict = {}
        sources: set = set()
        for c in self.chunks:
            ct = c["metadata"].get("content_type", "text")
            by_type[ct] = by_type.get(ct, 0) + 1
            sources.add(c["metadata"].get("source"))
        return {"chunks": len(self.chunks), "pdfs": len(self.manifest),
                "sources": len(sources), "by_type": by_type}
