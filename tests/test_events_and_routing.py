"""Offline tests for Marty's events feed, knowledge ingestion helpers and router intent."""
import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from events import feed  # noqa: E402
from rag import router  # noqa: E402

ICS = """BEGIN:VCALENDAR
VERSION:2.0
BEGIN:VEVENT
DTSTART:20261104T150000Z
DTEND:20261104T200000Z
SUMMARY:AUS x SUS Graduate and Professional Schools Fair
LOCATION:McGill University Centre\\, Room 301
DESCRIPTION:Meet law\\, med and grad programs.\\nBring
  questions!
URL:https://www.mcgill.ca/channels/event/grad-fair-1
END:VEVENT
END:VCALENDAR"""


def test_parse_ical_handles_utc_folding_and_escapes():
    (ev,) = feed.parse_ical(ICS)
    assert ev["title"].startswith("AUS x SUS Graduate")
    assert ev["location"] == "McGill University Centre, Room 301"
    assert "Bring questions!" in ev["description"]
    # 15:00Z in November = 10:00 Montréal (EST)
    assert ev["start"].hour == 10 and ev["end"].hour == 15


def test_page_text_fallback_parses_mcgill_header():
    start, end = feed.parse_event_page_text(
        "Event Macdonald Campus Career Fair 2026 Wednesday, January 21, 2026 11:00to15:00 Centennial Centre"
    )
    assert (start.year, start.month, start.day, start.hour) == (2026, 1, 21, 11)
    assert end.hour == 15


EVENTS = [
    {"title": "Grad & Professional Schools Fair", "start": "2026-11-04T10:00", "end": "2026-11-04T15:00",
     "tags": ["career"], "description": "", "url": "u1"},
    {"title": "Art Hive Open Studio", "start": "2026-11-05T11:00", "end": "2026-11-05T13:00",
     "tags": ["wellness"], "description": "", "url": "u2"},
    {"title": "Old event", "start": "2026-10-01T10:00", "end": "2026-10-01T11:00",
     "tags": ["career"], "description": "", "url": "u3"},
    {"title": "Macdonald Campus Career Fair", "start": "2027-01-20T11:00", "end": "2027-01-20T15:00",
     "tags": ["career"], "description": "", "url": "u4"},
]


def test_filter_drops_past_and_far_future_and_respects_tags():
    today = date(2026, 11, 2)
    got = feed.filter_events(EVENTS, days=14, today=today)
    assert [e["url"] for e in got] == ["u1", "u2"]
    got = feed.filter_events(EVENTS, days=14, tags={"career"}, today=today)
    assert [e["url"] for e in got] == ["u1"]
    got = feed.filter_events(EVENTS, days=90, query="career fair", today=today)
    assert [e["url"] for e in got] == ["u1", "u4"]


def test_format_for_prompt():
    out = feed.format_events_for_prompt(EVENTS[:1])
    assert "Wed 04 Nov 2026, 10:00–15:00" in out and "Grad & Professional" in out


@pytest.mark.parametrize("q,expected", [
    ("What career fairs are coming up?", True),
    ("what's on this week?", True),
    ("Is OAP happening?", True),
    ("Any networking events for finance?", True),
    ("How do I apply for a bursary?", False),
    ("What is the add/drop deadline?", False),
])
def test_events_intent(q, expected):
    assert router.is_events_query(q) is expected


def test_router_merges_events_with_vectorstore(monkeypatch):
    monkeypatch.setattr(feed, "get_events", lambda **kw: EVENTS[:2])
    monkeypatch.setattr(router, "_get_vectorstore", lambda: (lambda q, n_results=6: [
        {"text": "Career fairs need McGill ID.", "source_url": "https://www.mcgill.ca/caps/students/services/careerfairs"},
        {"text": "curated", "source_url": "marty://knowledge/x"},
    ]))
    ctx = router.route_query("What career events are on this week?")
    assert ctx["is_live"] is True
    assert "live feed" in ctx["text"] and "McGill ID" in ctx["text"]
    assert "u1" in ctx["sources"]
    assert not any(s.startswith("marty://") for s in ctx["sources"])


def test_refresh_cache_end_to_end_with_mocked_http(monkeypatch, tmp_path):
    listing = '<html><a href="/channels/event/grad-fair-1">Grad fair</a><a href="/channels/event/grad-fair-1">dup</a></html>'
    page = '<html><h1>Grad fair</h1><a href="/channels/node/123/ical">Add to calendar</a></html>'

    class R:
        def __init__(self, text): self.text = text

    def fake_get(url):
        if url.endswith("/ical"):
            return R(ICS)
        if "/channels/event/" in url:
            return R(page)
        return R(listing)

    monkeypatch.setattr(feed, "_get", fake_get)
    monkeypatch.setattr(feed, "CACHE_PATH", tmp_path / "cache.json")
    monkeypatch.setattr(feed, "REQUEST_DELAY", 0)
    events = feed.refresh_cache([("https://www.mcgill.ca/cle/events", "social")])
    assert len(events) == 1
    ev = events[0]
    assert ev["start"] == "2026-11-04T10:00"
    assert "career" in ev["tags"] and "social" in ev["tags"]
    assert json.loads((tmp_path / "cache.json").read_text())["events"][0]["title"] == ev["title"]


def test_knowledge_frontmatter_and_files():
    from rag.ingest import parse_frontmatter, KNOWLEDGE_DIR
    files = sorted(KNOWLEDGE_DIR.glob("*.md"))
    assert len(files) >= 5
    for f in files:
        meta, body = parse_frontmatter(f.read_text())
        assert meta.get("source_url", "").startswith("http"), f.name
        assert meta.get("verified"), f.name
        assert body.lstrip().startswith("#"), f.name
