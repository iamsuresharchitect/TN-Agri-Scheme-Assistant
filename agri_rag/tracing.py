"""Optional LangSmith tracing.

When LANGSMITH_TRACING=true and LANGSMITH_API_KEY is set, every farmer question is
recorded in LangSmith as one trace:

    TN Agri RAG (chain)
      ├── FAISS retrieval (retriever)  - retrieved chunks + similarity scores
      └── Claude call (llm)            - prompt, answer, tokens, latency

When tracing is off (or the langsmith package is missing) these helpers do nothing,
so the app runs exactly the same without a LangSmith account.
"""
import functools
import logging

from . import config

log = logging.getLogger(__name__)

ENABLED = bool(config.LANGSMITH_TRACING and config.LANGSMITH_API_KEY)
_traceable = _wrap_anthropic = _Client = None

if ENABLED:
    try:
        from langsmith import Client as _Client
        from langsmith import traceable as _traceable
        from langsmith.wrappers import wrap_anthropic as _wrap_anthropic

        log.info("LangSmith tracing enabled (project: %s)", config.LANGSMITH_PROJECT)
    except ImportError:
        log.warning("LANGSMITH_TRACING is on but the 'langsmith' package is not installed. "
                    "Run: pip install langsmith")
        ENABLED = False
elif config.LANGSMITH_TRACING:
    log.warning("LANGSMITH_TRACING is on but LANGSMITH_API_KEY is empty - tracing disabled.")


def traceable(**kwargs):
    """Like langsmith.traceable, but a no-op (that drops `langsmith_extra`) when tracing is off."""

    def decorator(fn):
        if ENABLED:
            return _traceable(**kwargs)(fn)

        @functools.wraps(fn)
        def passthrough(*args, langsmith_extra=None, **kw):
            return fn(*args, **kw)

        return passthrough

    return decorator


def wrap_client(client):
    """Wrap the Anthropic client so each Claude call appears as an LLM run."""
    if not ENABLED:
        return client
    try:
        return _wrap_anthropic(client)
    except Exception as exc:
        log.warning("Could not wrap Anthropic client for LangSmith: %s", exc)
        return client


_client_cache: dict = {}


def send_feedback(run_id: str, score: int, comment: str | None = None) -> bool:
    """Attach a farmer's 👍 (1) / 👎 (0) rating to a trace. Returns True if sent."""
    if not ENABLED or not run_id:
        return False
    try:
        if "client" not in _client_cache:
            _client_cache["client"] = _Client()
        _client_cache["client"].create_feedback(run_id, key="farmer_rating", score=score, comment=comment)
        return True
    except Exception as exc:
        log.warning("Could not send feedback to LangSmith: %s", exc)
        return False
