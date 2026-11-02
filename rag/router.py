#!/usr/bin/env python3
"""
Marty.AI — Hybrid Query Router
Routes student questions to either the vector store or a live web fetch.

Usage (standalone test):
    python rag/router.py "What are the registration deadlines this semester?"
    python rag/router.py "How do I apply for financial aid?"

Exported API (used by backend/main.py):
    route_query(question, n_results=6) -> dict
    is_time_sensitive(query) -> bool
"""
import json
import logging
import re
import sys
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

sys.path.insert(0, str(Path(__file__).parent.parent))

SCRAPED_DATA_DIR = Path(__file__).parent.parent / "scraper" / "scraped_data"
LIVE_FETCH_TIMEOUT = 10       # seconds
LIVE_FETCH_MAX_CHARS = 4000   # truncate live text to keep prompts reasonable
USER_AGENT = "MartyAIBot/1.0 (educational)"

# Regex patterns that signal the answer may change day-to-day
_TS_PATTERNS = [
    r"\bdeadline\b",
    r"\bwhen\b.{0,20}\b(open|close|due|start|end|available)\b",
    r"\bfee\b",
    r"\bcost\b",
    r"\btuition\b",
    r"\bthis\s+semester\b",
    r"\bnext\s+semester\b",
    r"\btoday\b",
    r"\bcurrently\b",
    r"\bopen\s+now\b",
    r"\bstill\s+open\b",
    r"\bstill\s+closed\b",
    r"\bregistration\s+(open|close|date|deadline)\b",
    r"\bscholarship\s+deadline\b",
    r"\baward\s+deadline\b",
    r"\bimportant\s+dates\b",
    r"\badd.drop\b",
    r"\bwithdrawal\s+deadline\b",
]
_TS_RE = re.compile("|".join(_TS_PATTERNS), re.IGNORECASE)

log = logging.getLogger(__name__)


# ── helpers ───────────────────────────────────────────────────────────────────

def is_time_sensitive(query: str) -> bool:
    """Return True if the query is likely asking about time-sensitive information."""
    return bool(_TS_RE.search(query))


# Questions about what's on — events, fairs, workshops, things to do.
_EVENTS_PATTERNS = [
    r"\bevents?\b",
    r"\bfairs?\b",
    r"\bworkshops?\b",
    r"\binfo(rmation)?\s+sessions?\b",
    r"\bnetworking\b",
    r"\bactivities\s+night\b",
    r"\bwhat'?s\s+(on|happening|going\s+on)\b",
    r"\bhappening\b",
    r"\b(this|next)\s+(week|weekend|month)\b",
    r"\btonight\b",
    r"\btomorrow\b",
    r"\bupcoming\b",
    r"\bthings\s+to\s+do\b",
    r"\bpanels?\b",
    r"\bspeakers?\b",
    r"\boap\b|\bopen\s+air\s+pub\b",
]
_EVENTS_RE = re.compile("|".join(_EVENTS_PATTERNS), re.IGNORECASE)

_EVENT_TAG_HINTS = {
    "career": r"career|job|recruit|employer|internship|network|fair|cv|resume|interview|finance|consult",
    "clubs": r"club|activities night|ssmu|involved|society|volunteer",
    "wellness": r"wellness|mental|stress|support|yoga|mindful",
    "social": r"social|party|fun|meet people|friends|oap|pub",
    "academic": r"lecture|talk|seminar|panel|speaker",
}


def is_events_query(query: str) -> bool:
    """Return True if the student is asking what's on / upcoming events."""
    return bool(_EVENTS_RE.search(query))


def _event_tags_for(query: str) -> Optional[set[str]]:
    tags = {t for t, pat in _EVENT_TAG_HINTS.items() if re.search(pat, query, re.I)}
    return tags or None


def _events_context(question: str) -> Optional[dict]:
    """Pull upcoming events from the live feed cache (never blocks on a crawl)."""
    try:
        from events.feed import get_events, format_events_for_prompt
    except Exception as exc:  # pragma: no cover
        log.warning("Events feed unavailable: %s", exc)
        return None
    days = 30 if re.search(r"month|semester|term", question, re.I) else 14
    events = get_events(days=days, tags=_event_tags_for(question), allow_refresh=False)
    if not events:
        events = get_events(days=days, allow_refresh=False)  # broaden before giving up
    if not events:
        return None
    return {
        "text": "Upcoming public McGill events (live feed):\n" + format_events_for_prompt(events),
        "sources": [e["url"] for e in events[:8] if e.get("url")],
    }


def _load_sensitive_urls() -> list[dict]:
    manifest = SCRAPED_DATA_DIR / "_index.json"
    if not manifest.exists():
        return []
    index: list[dict] = json.loads(manifest.read_text(encoding="utf-8"))
    return [e for e in index if e.get("time_sensitive")]


def _score_url(url: str, query_words: set[str]) -> int:
    url_words = set(re.findall(r"\w+", url.lower()))
    return len(query_words & url_words)


def _pick_live_url(query: str) -> str:
    """Choose the most relevant time-sensitive URL from the manifest."""
    candidates = _load_sensitive_urls()
    if not candidates:
        # Hard-coded fallback for when no manifest exists yet
        return "https://www.mcgill.ca/registrar/registration"

    query_words = set(re.findall(r"\w+", query.lower()))
    best = max(candidates, key=lambda e: _score_url(e["url"], query_words))
    return best["url"]


def _live_fetch(url: str) -> Optional[str]:
    """Fetch a live page and return cleaned plain text."""
    try:
        resp = requests.get(
            url,
            timeout=LIVE_FETCH_TIMEOUT,
            headers={"User-Agent": USER_AGENT},
        )
        resp.raise_for_status()
    except requests.RequestException as exc:
        log.warning("Live fetch failed for %s: %s", url, exc)
        return None

    soup = BeautifulSoup(resp.text, "lxml")
    for tag in soup.find_all(["nav", "footer", "script", "style", "noscript", "header"]):
        tag.decompose()

    main = (
        soup.find("main")
        or soup.find(id=re.compile(r"^(main|content)$", re.I))
        or soup.body
    )
    text = (main or soup).get_text("\n", strip=True)
    # Collapse whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ── vector store (lazy import to avoid circular deps) ─────────────────────────

_query_vectorstore = None


def _get_vectorstore():
    global _query_vectorstore
    if _query_vectorstore is None:
        from rag.ingest import query_vectorstore as _qvs
        _query_vectorstore = _qvs
    return _query_vectorstore


# ── public API ────────────────────────────────────────────────────────────────

def route_query(question: str, n_results: int = 6) -> dict:
    """
    Route *question* to the appropriate retrieval path and return a unified context dict:

        {
            "text":    str,        # retrieved context passages (may be empty)
            "sources": list[str],  # deduplicated source URLs
            "is_live": bool,       # True when fetched live from mcgill.ca
        }
    """
    events_ctx = _events_context(question) if is_events_query(question) else None

    if is_time_sensitive(question) and not events_ctx:
        log.info("Time-sensitive query — attempting live fetch")
        url = _pick_live_url(question)
        text = _live_fetch(url)
        if text:
            log.info("Live fetch succeeded: %s", url)
            return {
                "text": text[:LIVE_FETCH_MAX_CHARS],
                "sources": [url],
                "is_live": True,
            }
        log.warning("Live fetch failed — falling back to vector store")

    # General path: vector store (plus the live events feed when relevant)
    qvs = _get_vectorstore()
    try:
        results = qvs(question, n_results=n_results)
    except Exception as exc:
        log.warning("Vector store query failed: %s", exc)
        results = []

    parts: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()

    if events_ctx:
        parts.append(events_ctx["text"])
        for src in events_ctx["sources"]:
            if src not in seen:
                sources.append(src)
                seen.add(src)

    if not results and not parts:
        return {"text": "", "sources": [], "is_live": False}

    for r in results:
        parts.append(r["text"])
        src = r.get("source_url", "")
        if src and src not in seen:
            sources.append(src)
            seen.add(src)

    return {
        "text": "\n\n---\n\n".join(parts),
        "sources": [s for s in sources if not s.startswith("marty://")],
        "is_live": bool(events_ctx),
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).parent.parent / ".env")

    parser = argparse.ArgumentParser(description="Test the Marty.AI query router")
    parser.add_argument("query", nargs="?", default="What are the course add/drop deadlines?")
    args = parser.parse_args()

    q = args.query
    print(f"\nQuery       : {q!r}")
    print(f"Sensitive?  : {is_time_sensitive(q)}")
    result = route_query(q)
    print(f"Is live     : {result['is_live']}")
    print(f"Sources     : {result['sources']}")
    print("\n── Context (first 600 chars) ──────────────────────────────────────────")
    print(result["text"][:600])
