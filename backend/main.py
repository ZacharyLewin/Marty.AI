#!/usr/bin/env python3
"""
AI'dvisor McGill Advisor — FastAPI Backend

Endpoints:
    POST /chat/stream   Server-Sent Events streaming response
    POST /chat          Non-streaming fallback
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
from pathlib import Path
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

# ── model & prompt ────────────────────────────────────────────────────────────

CLAUDE_MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 1024

SYSTEM_PROMPT = """You are AI'dvisor, a warm, knowledgeable, and empathetic student advisor for McGill University. \
Your mission is to help McGill students navigate every aspect of academic life — \
registration, financial aid, wellness, housing, international student requirements, \
academic deadlines, course selection, and more.

## How to respond
- Be friendly, encouraging, and clear. Students may be anxious or under pressure.
- Ground every specific claim (dates, fees, eligibility rules) in the provided context.
- When quoting facts, mention the source so the student can follow up.
- If the context is insufficient to answer confidently, say so plainly and direct the \
  student to the appropriate McGill office — never fabricate information.
- For questions about deadlines, fees, or registration status, remind the student that \
  details change each semester and to verify with the official source.

## Always append to responses involving deadlines, requirements, or eligibility:
> ⚠️ Please verify this information directly with your McGill advisor or the relevant office.

## Emergency & wellness contacts (include when relevant)
- **McGill Security (24/7):** 514-398-3000
- **Student Wellness Hub:** 514-398-6017
- **Nightline (peer support, evenings):** 514-398-6246
- **Campus Life & Engagement:** https://www.mcgill.ca/studentservices/

Never invent contact information, course codes, or policy details not found in the context."""

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
    title="AI'dvisor McGill Advisor API",
    version="1.0.0",
    description="AI'dvisor — AI-powered student advisor for McGill University",
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


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        context = route_query(req.message)
        messages = build_messages(req.history, req.message, context)
        client = get_anthropic()
        resp = client.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
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
                system=SYSTEM_PROMPT,
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
