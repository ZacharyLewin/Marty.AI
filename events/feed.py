#!/usr/bin/env python3
"""
Marty.AI — Live McGill events feed

CaPS moved its own events calendar into myFuture (login-only), but McGill's
faculty, Student Services and campus-life sites still publish public event
listings. Each event page carries an "Add to calendar" iCal link, which gives
clean, structured dates. This module:

  1. reads the public listing pages in EVENT_SOURCES,
  2. follows links to individual event pages (…/channels/event/…),
  3. parses each event from its iCal link (falling back to the page text),
  4. tags it (career / clubs / wellness / social / academic), and
  5. caches everything in events/events_cache.json.

Usage:
    python events/feed.py                 # refresh cache, print next 14 days
    python events/feed.py --days 30 --q "career fair"

Exported API (used by rag/router.py and backend/main.py):
    get_events(days=14, query=None, tags=None, max_age_hours=6) -> list[dict]
    format_events_for_prompt(events) -> str
    refresh_cache() -> list[dict]
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

try:
    from zoneinfo import ZoneInfo
    MONTREAL = ZoneInfo("America/Toronto")
except Exception:  # pragma: no cover
    MONTREAL = None

log = logging.getLogger(__name__)

CACHE_PATH = Path(__file__).parent / "events_cache.json"
USER_AGENT = "MartyAIBot/1.0 (McGill student project; educational)"
REQUEST_TIMEOUT = 12
REQUEST_DELAY = 1.0
MAX_EVENTS_PER_SOURCE = 40

# Public McGill pages that list upcoming events. Each entry is (url, default_tag).
# A source that 404s or changes layout is skipped, never fatal.
EVENT_SOURCES: list[tuple[str, str]] = [
    ("https://www.mcgill.ca/studentservices/events", "wellness"),
    ("https://www.mcgill.ca/cle/events", "social"),
    ("https://www.mcgill.ca/desautels/events", "career"),
    ("https://www.mcgill.ca/careers4engineers/events", "career"),
    ("https://www.mcgill.ca/channels/tag/Soutar%20Career%20Centre/channel_event", "career"),
    ("https://www.mcgill.ca/caps/events", "career"),
    ("https://www.mcgill.ca/arts/events", "academic"),
    ("https://www.mcgill.ca/science/events", "academic"),
    ("https://www.mcgill.ca/internationalstudents/events", "wellness"),
]

TAG_KEYWORDS: dict[str, list[str]] = {
    "career": [
        "career", "fair", "recruit", "employer", "info session", "information session",
        "networking", "internship", "resume", "cv", "cover letter", "interview",
        "job", "alumni", "mentor", "linkedin", "caps", "soutar", "techfair", "meet mcgill",
    ],
    "clubs": ["activities night", "club", "ssmu", "society", "volunteer", "student group"],
    "wellness": ["wellness", "mental health", "workshop", "mindful", "yoga", "peer", "support", "art hive"],
    "social": ["social", "party", "pub", "oap", "frosh", "orientation", "tour", "festival", "games"],
    "academic": ["lecture", "seminar", "talk", "colloquium", "symposium", "conference", "panel", "reading"],
}

_EVENT_LINK_RE = re.compile(r"/channels/event/|/channels/node/\d+|/event/", re.I)
_HEADER_DATE_RE = re.compile(
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+"
    r"([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})\s*(\d{1,2}:\d{2})?\s*(?:to\s*(\d{1,2}:\d{2}))?"
)


# ── fetching ──────────────────────────────────────────────────────────────────

_session: Optional[requests.Session] = None


def _get(url: str) -> Optional[requests.Response]:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers.update({"User-Agent": USER_AGENT})
    try:
        resp = _session.get(url, timeout=REQUEST_TIMEOUT, allow_redirects=True)
        resp.raise_for_status()
        return resp
    except requests.RequestException as exc:
        log.warning("Events fetch failed for %s: %s", url, exc)
        return None


# ── iCal parsing (tiny RFC 5545 subset, no extra dependency) ─────────────────

def _unfold(ics: str) -> list[str]:
    lines: list[str] = []
    for raw in ics.replace("\r\n", "\n").split("\n"):
        if raw.startswith((" ", "\t")) and lines:
            lines[-1] += raw[1:]
        else:
            lines.append(raw)
    return lines


def _unescape(value: str) -> str:
    return (value.replace("\\n", "\n").replace("\\N", "\n")
                 .replace("\\,", ",").replace("\\;", ";").replace("\\\\", "\\"))


def _parse_ical_dt(value: str, params: str) -> Optional[datetime]:
    value = value.strip()
    try:
        if "VALUE=DATE" in params and len(value) == 8:
            return datetime.strptime(value, "%Y%m%d")
        if value.endswith("Z"):
            dt = datetime.strptime(value, "%Y%m%dT%H%M%SZ")
            if MONTREAL is not None:
                from datetime import timezone
                dt = dt.replace(tzinfo=timezone.utc).astimezone(MONTREAL).replace(tzinfo=None)
            return dt
        return datetime.strptime(value[:15], "%Y%m%dT%H%M%S")
    except ValueError:
        return None


def parse_ical(ics: str) -> list[dict]:
    """Return a list of {title, start, end, location, url, description} from an iCal string."""
    events: list[dict] = []
    current: Optional[dict] = None
    for line in _unfold(ics):
        if line == "BEGIN:VEVENT":
            current = {}
        elif line == "END:VEVENT":
            if current is not None and current.get("start"):
                events.append(current)
            current = None
        elif current is not None and ":" in line:
            key_part, _, value = line.partition(":")
            key, _, params = key_part.partition(";")
            key = key.upper()
            if key == "SUMMARY":
                current["title"] = _unescape(value).strip()
            elif key == "DTSTART":
                current["start"] = _parse_ical_dt(value, params)
            elif key == "DTEND":
                current["end"] = _parse_ical_dt(value, params)
            elif key == "LOCATION":
                current["location"] = _unescape(value).strip()
            elif key == "URL":
                current["url"] = value.strip()
            elif key == "DESCRIPTION":
                current["description"] = _unescape(value).strip()
    return events


# ── HTML fallback ─────────────────────────────────────────────────────────────

def parse_event_page_text(text: str) -> tuple[Optional[datetime], Optional[datetime]]:
    """Parse McGill's 'Wednesday, January 21, 2026 11:00to15:00' header line."""
    m = _HEADER_DATE_RE.search(text)
    if not m:
        return None, None
    _, month, day, year, t1, t2 = m.groups()
    try:
        base = datetime.strptime(f"{month} {day} {year}", "%B %d %Y")
    except ValueError:
        return None, None
    start = end = base
    if t1:
        h, mi = map(int, t1.split(":"))
        start = base.replace(hour=h, minute=mi)
    if t2:
        h, mi = map(int, t2.split(":"))
        end = base.replace(hour=h, minute=mi)
    return start, end


def _event_from_page(url: str, default_tag: str) -> Optional[dict]:
    resp = _get(url)
    if resp is None:
        return None
    soup = BeautifulSoup(resp.text, "lxml")

    # 1) Preferred: the page's "Add to calendar" iCal link
    ical_link = soup.find("a", href=re.compile(r"/ical/?$"))
    if ical_link:
        ics_resp = _get(urljoin(url, ical_link["href"]))
        if ics_resp is not None:
            parsed = parse_ical(ics_resp.text)
            if parsed:
                ev = parsed[0]
                ev.setdefault("url", url)
                ev["url"] = url  # canonical human page
                return _finish(ev, default_tag)

    # 2) Fallback: page title + header date line
    main = soup.find("main") or soup.body or soup
    text = main.get_text(" ", strip=True)
    start, end = parse_event_page_text(text)
    if not start:
        return None
    h1 = soup.find("h1")
    title = h1.get_text(" ", strip=True) if h1 else (soup.title.get_text(strip=True) if soup.title else url)
    loc_match = re.search(r"\d{1,5}[^,]{2,60},\s*(?:Montr[eé]al|St[e]?-?Anne[^,]*|Ste-Anne[^,]*)[^.]{0,40}", text)
    return _finish({
        "title": title,
        "start": start,
        "end": end,
        "location": loc_match.group(0).strip() if loc_match else "",
        "url": url,
        "description": text[:600],
    }, default_tag)


def _tag(ev: dict, default_tag: str) -> list[str]:
    hay = f"{ev.get('title', '')} {ev.get('description', '')}".lower()
    tags = {t for t, kws in TAG_KEYWORDS.items() if any(k in hay for k in kws)}
    tags.add(default_tag)
    return sorted(tags)


def _finish(ev: dict, default_tag: str) -> dict:
    ev["tags"] = _tag(ev, default_tag)
    ev["description"] = (ev.get("description") or "")[:600]
    for k in ("start", "end"):
        if isinstance(ev.get(k), datetime):
            ev[k] = ev[k].isoformat(timespec="minutes")
    ev.setdefault("location", "")
    return ev


def _event_links(listing_url: str) -> list[str]:
    resp = _get(listing_url)
    if resp is None:
        return []
    soup = BeautifulSoup(resp.text, "lxml")
    seen: set[str] = set()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        full = urljoin(listing_url, a["href"]).split("#")[0]
        if "mcgill.ca" not in urlparse(full).netloc:
            continue
        if _EVENT_LINK_RE.search(full) and not full.endswith("/ical") and full not in seen:
            seen.add(full)
            links.append(full)
    return links[:MAX_EVENTS_PER_SOURCE]


# ── cache ─────────────────────────────────────────────────────────────────────

def refresh_cache(sources: Iterable[tuple[str, str]] = EVENT_SOURCES) -> list[dict]:
    """Crawl all event sources and rewrite the cache. Returns the events."""
    by_key: dict[tuple, dict] = {}
    for listing_url, tag in sources:
        links = _event_links(listing_url)
        log.info("Events: %s -> %d event links", listing_url, len(links))
        for link in links:
            ev = _event_from_page(link, tag)
            time.sleep(REQUEST_DELAY)
            if not ev:
                continue
            key = (ev.get("title", "").lower(), ev.get("start", "")[:10])
            if key in by_key:
                by_key[key]["tags"] = sorted(set(by_key[key]["tags"]) | set(ev["tags"]))
            else:
                ev["source_listing"] = listing_url
                by_key[key] = ev
    events = sorted(by_key.values(), key=lambda e: e.get("start", ""))
    CACHE_PATH.write_text(json.dumps({
        "fetched_at": datetime.now().isoformat(timespec="seconds"),
        "events": events,
    }, indent=2), encoding="utf-8")
    log.info("Events cache written: %d events -> %s", len(events), CACHE_PATH)
    return events


def _load_cache() -> tuple[Optional[datetime], list[dict]]:
    if not CACHE_PATH.exists():
        return None, []
    try:
        data = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        fetched = datetime.fromisoformat(data["fetched_at"])
        return fetched, data.get("events", [])
    except Exception as exc:  # corrupt cache → treat as empty
        log.warning("Could not read events cache: %s", exc)
        return None, []


def _today() -> date:
    if MONTREAL is not None:
        return datetime.now(MONTREAL).date()
    return date.today()


def filter_events(
    events: list[dict],
    days: int = 14,
    query: Optional[str] = None,
    tags: Optional[set[str]] = None,
    today: Optional[date] = None,
) -> list[dict]:
    today = today or _today()
    horizon = today + timedelta(days=days)
    words = [w for w in re.findall(r"\w+", (query or "").lower()) if len(w) > 2]
    out: list[dict] = []
    for ev in events:
        try:
            start = datetime.fromisoformat(ev["start"]).date()
            end = datetime.fromisoformat(ev.get("end") or ev["start"]).date()
        except (KeyError, ValueError):
            continue
        if end < today or start > horizon:
            continue
        if tags and not (set(ev.get("tags", [])) & tags):
            continue
        if words:
            hay = f"{ev.get('title', '')} {ev.get('description', '')} {' '.join(ev.get('tags', []))}".lower()
            if not any(w in hay for w in words):
                continue
        out.append(ev)
    return out


def get_events(
    days: int = 14,
    query: Optional[str] = None,
    tags: Optional[set[str]] = None,
    max_age_hours: float = 6,
    allow_refresh: bool = True,
) -> list[dict]:
    """Return upcoming events, refreshing the cache if it is stale."""
    fetched, events = _load_cache()
    stale = fetched is None or datetime.now() - fetched > timedelta(hours=max_age_hours)
    if stale and allow_refresh:
        try:
            events = refresh_cache()
        except Exception as exc:
            log.warning("Events refresh failed, using cached copy: %s", exc)
    return filter_events(events, days=days, query=query, tags=tags)


def format_events_for_prompt(events: list[dict], limit: int = 15) -> str:
    if not events:
        return ""
    lines = []
    for ev in events[:limit]:
        start = datetime.fromisoformat(ev["start"])
        when = start.strftime("%a %d %b %Y")
        if start.hour or start.minute:
            when += start.strftime(", %H:%M")
            if ev.get("end"):
                end = datetime.fromisoformat(ev["end"])
                if end.date() == start.date() and end > start:
                    when += end.strftime("–%H:%M")
        loc = f" @ {ev['location']}" if ev.get("location") else ""
        tags = ", ".join(ev.get("tags", []))
        lines.append(f"- {when}: {ev.get('title', 'Untitled')}{loc} [{tags}] — {ev.get('url', '')}")
    return "\n".join(lines)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="Refresh and show Marty's McGill events feed")
    p.add_argument("--days", type=int, default=14)
    p.add_argument("--q", default=None, help="keyword filter, e.g. 'career fair'")
    p.add_argument("--no-refresh", action="store_true", help="read cache only")
    args = p.parse_args()
    if not args.no_refresh:
        refresh_cache()
    _, cached = _load_cache()
    upcoming = filter_events(cached, days=args.days, query=args.q)
    print(format_events_for_prompt(upcoming, limit=100) or "No upcoming events found.")
