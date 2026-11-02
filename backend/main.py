#!/usr/bin/env python3
"""
Marty.AI — FastAPI Backend

Endpoints:
    POST /chat/stream   Server-Sent Events streaming response
    POST /chat          Non-streaming fallback
    GET  /events        Upcoming public McGill events (live feed)
    GET  /health        Liveness check

Usage:
    python backend/main.py
    # or with uvicorn directly:
    uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
"""
import asyncio
import json
import logging
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
from typing import Optional

# Make sibling packages importable when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import anthropic
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from rag.router import route_query
from events.feed import get_events, refresh_cache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── model & prompt ────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT_TEMPLATE = """You are Marty — McGill's unofficial AI advisor, named after the Martlet. \
Think of yourself as the upper-year friend everyone wishes they'd met in U0: you've been \
through add/drop panic, queued for OAP up the Y, survived finals in McLennan, and you know \
which office actually answers the phone. You help McGill students with academics, \
registration, financial aid, wellness, international-student questions, and — just as much — \
careers, career fairs, clubs, events, and the culture of campus life.

Today is {today} (Montréal time). Use this to judge what is upcoming vs. already past.

## Voice
- Warm, direct, a little witty. Talk like a student, not a brochure. Short paragraphs; \
bullets when listing options.
- Use McGill vocabulary naturally when it fits (U1/U2/U3, add/drop, Leacock, Shatner/the \
SSMU building, the Y, Lower Field, Milton-Parc, Gerts, OAP, Frosh, Activities Night, \
Redbirds/Martlets). Never force it and never let flavour replace facts.
- If someone writes in French, answer in French.
- Be encouraging. Students asking about careers are often anxious; give them a concrete \
next step (an event, an office, a club, a deadline) rather than generic advice.

## Accuracy rules (non-negotiable)
- Ground every specific claim (dates, times, locations, fees, eligibility) in the retrieved \
context and name the source so the student can check it.
- If an event in the context is already past relative to today, say so; don't present it \
as upcoming.
- Marty cannot see Minerva, myCourses, or myFuture (all behind McGill login). When the full \
answer lives there — CaPS workshops, job postings, advising bookings, grades, registration \
status — say so plainly and tell the student exactly where to click.
- If the context is thin, say what you don't know and point to the right office. Never \
invent contact details, course codes, dates, or policies.

## When answering about deadlines, requirements, or eligibility, end with:
> ⚠️ Double-check with the official page or office — details change each term.

## Wellness & emergency contacts (include when relevant)
- **McGill Security (24/7):** 514-398-3000
- **Student Wellness Hub:** 514-398-6017
- **Nightline (peer support, evenings):** 514-398-6246
- **Student Services:** https://www.mcgill.ca/studentservices/"""


def system_prompt() -> str:
    """System prompt with today's date filled in (Marty must know what 'upcoming' means)."""
    now = datetime.now(ZoneInfo("America/Toronto"))
    return SYSTEM_PROMPT_TEMPLATE.format(today=now.strftime("%A %d %B %Y"))

# ── CORS ──────────────────────────────────────────────────────────────────────

_raw_origins = [
    "http://localhost:3000",
    "http://localhost:8080",
    "http://localhost:5500",
    "http://127.0.0.1:3000",
    "http://127.0.0.1:8080",
    "http://127.0.0.1:5500",
    os.getenv("FRONTEND_URL", ""),
]
ALLOWED_ORIGINS = [o for o in _raw_origins if o]

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Marty.AI API",
    version="1.1.0",
    description="Marty — AI-powered student advisor for McGill University",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Anthropic client (lazy) ───────────────────────────────────────────────────

_anthropic_client: Optional[anthropic.Anthropic] = None


def get_anthropic() -> anthropic.Anthropic:
    global _anthropic_client
    if _anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set in environment")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


# ── Pydantic models ───────────────────────────────────────────────────────────

class Message(BaseModel):
    role: str                   # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    history: list[Message] = Field(default_factory=list)


class ChatResponse(BaseModel):
    reply: str
    sources: list[str]
    is_live: bool


# ── message builder ───────────────────────────────────────────────────────────

def build_messages(history: list[Message], user_message: str, context: dict) -> list[dict]:
    """
    Prepend retrieved context to the latest user message and
    convert to the format Anthropic's API expects.
    """
    context_block = ""
    if context["text"]:
        freshness = " (fetched live)" if context["is_live"] else " (from knowledge base)"
        sources_md = "\n".join(f"- {s}" for s in context["sources"])
        context_block = (
            f"<retrieved_context{freshness}>\n"
            f"{context['text']}\n\n"
            f"Sources:\n{sources_md}\n"
            f"</retrieved_context>\n\n"
        )

    messages = [{"role": m.role, "content": m.content} for m in history]
    messages.append({"role": "user", "content": context_block + user_message})
    return messages


# ── endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": CLAUDE_MODEL,
        "version": app.version,
    }


# ── live events feed: refreshed in the background, never inside a chat request ─

EVENTS_REFRESH_HOURS = float(os.getenv("EVENTS_REFRESH_HOURS", "6"))


def _events_refresher() -> None:
    while True:
        try:
            refresh_cache()
        except Exception as exc:
            log.warning("Background events refresh failed: %s", exc)
        time.sleep(EVENTS_REFRESH_HOURS * 3600)


@app.on_event("startup")
def _start_events_refresher() -> None:
    if os.getenv("EVENTS_FEED", "on").lower() != "off":
        threading.Thread(target=_events_refresher, daemon=True, name="events-refresh").start()


@app.get("/events")
async def events(days: int = 14, q: Optional[str] = None, tag: Optional[str] = None):
    """Upcoming public McGill events, e.g. /events?days=7&tag=career"""
    days = max(1, min(days, 90))
    tags = {tag} if tag else None
    items = get_events(days=days, query=q, tags=tags, allow_refresh=False)
    return {"count": len(items), "events": items}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        context = route_query(req.message)
        messages = build_messages(req.history, req.message, context)
        client = get_anthropic()
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=system_prompt(),
            messages=messages,
        )
        reply = resp.content[0].text
        return ChatResponse(reply=reply, sources=context["sources"], is_live=context["is_live"])
    except Exception as exc:
        log.exception("Error in /chat: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Server-Sent Events streaming endpoint.

    Event types:
        meta   — JSON with {sources: [...], is_live: bool}  (sent first)
        text   — JSON with {text: "..."} (one per streamed chunk)
        error  — JSON with {message: "..."} (on failure)
        done   — empty payload (stream finished)
    """
    try:
        context = route_query(req.message)
        messages = build_messages(req.history, req.message, context)
    except Exception as exc:
        log.exception("Router error: %s", exc)
        async def error_stream():
            payload = json.dumps({"message": str(exc)})
            yield f"event: error\ndata: {payload}\n\n"
        return StreamingResponse(error_stream(), media_type="text/event-stream")

    async def event_generator():
        # 1. Send metadata (sources, is_live) before any text
        meta = json.dumps({"sources": context["sources"], "is_live": context["is_live"]})
        yield f"event: meta\ndata: {meta}\n\n"

        try:
            client = get_anthropic()
            with client.messages.stream(
                model=CLAUDE_MODEL,
                max_tokens=MAX_TOKENS,
                system=system_prompt(),
                messages=messages,
            ) as stream:
                for chunk in stream.text_stream:
                    payload = json.dumps({"text": chunk})
                    yield f"event: text\ndata: {payload}\n\n"
                    await asyncio.sleep(0)   # yield control to the event loop
        except Exception as exc:
            log.exception("Stream error: %s", exc)
            payload = json.dumps({"message": str(exc)})
            yield f"event: error\ndata: {payload}\n\n"
            return

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # disable Nginx buffering
            "Connection": "keep-alive",
        },
    )


# ── entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
