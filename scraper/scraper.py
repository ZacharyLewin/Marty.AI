#!/usr/bin/env python3
"""
AI'dvisor McGill Advisor — Web Scraper
Crawls mcgill.ca and saves clean Markdown files + an _index.json manifest.

Usage:
    python scraper/scraper.py                   # full crawl (200 pages)
    python scraper/scraper.py --quick           # seed URLs only (testing)
    python scraper/scraper.py --max-pages 20    # limited crawl
"""
import argparse
import hashlib
import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup, NavigableString

# ── configuration ─────────────────────────────────────────────────────────────

SEED_URLS: list[str] = [
    "https://www.mcgill.ca/students/",
    "https://www.mcgill.ca/study/",
    "https://www.mcgill.ca/registrar/",
    "https://www.mcgill.ca/studentaid/",
    "https://www.mcgill.ca/caps/",
    "https://www.mcgill.ca/internationalstudents/",
    # Faculty advising
    "https://www.mcgill.ca/arts/students/",
    "https://www.mcgill.ca/science/students/",
    "https://www.mcgill.ca/engineering/students/",
    "https://www.mcgill.ca/desautels/programs/",
]

# URL path patterns that indicate time-sensitive content
TIME_SENSITIVE_PATTERNS: list[str] = [
    r"deadline",
    r"registration",
    r"fees",
    r"tuition",
    r"calendar",
    r"important.dates",
    r"add.drop",
    r"withdrawal",
    r"scholarship",
    r"bursary",
    r"financial.aid",
    r"award",
    r"schedule",
    r"key.dates",
]
_TS_RE = re.compile("|".join(TIME_SENSITIVE_PATTERNS), re.IGNORECASE)

MAX_PAGES = 200
REQUEST_DELAY = 1.5          # seconds between fetches
MAX_LINKS_PER_PAGE = 40      # guard against link explosion
ALLOWED_DOMAIN = "mcgill.ca"
USER_AGENT = "AIdvisorBot/1.0 (educational research; not-for-profit)"
OUTPUT_DIR = Path(__file__).parent / "scraped_data"

# Tags that are pure chrome and add no informational value
_JUNK_TAGS = [
    "nav", "footer", "script", "style", "noscript", "header",
    "aside", "iframe", "form", "button", "svg", "canvas",
]
_JUNK_CLASS_RE = re.compile(
    r"breadcrumb|cookie|banner|advertisement|sidebar|social|share|"
    r"related|comment|popup|modal|overlay|menu|navbar|topbar",
    re.IGNORECASE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)


# ── robots.txt ────────────────────────────────────────────────────────────────

_robot_cache: dict[str, RobotFileParser] = {}


def _robot_parser(base_url: str) -> RobotFileParser:
    if base_url not in _robot_cache:
        rp = RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        try:
            rp.read()
        except Exception:
            pass  # if robots.txt is unreachable, allow crawling
        _robot_cache[base_url] = rp
    return _robot_cache[base_url]


def can_fetch(url: str) -> bool:
    p = urlparse(url)
    base = f"{p.scheme}://{p.netloc}"
    return _robot_parser(base).can_fetch(USER_AGENT, url)


# ── HTML → Markdown ───────────────────────────────────────────────────────────

def _strip_junk(soup: BeautifulSoup) -> None:
    for tag in soup.find_all(_JUNK_TAGS):
        tag.decompose()
    for el in soup.find_all(class_=_JUNK_CLASS_RE):
        el.decompose()
    for el in soup.find_all(id=_JUNK_CLASS_RE):
        el.decompose()


def _find_main(soup: BeautifulSoup):
    return (
        soup.find("main")
        or soup.find(id=re.compile(r"^(main|content|main.content|page.content)$", re.I))
        or soup.find(class_=re.compile(r"main.content|page.content|entry.content", re.I))
        or soup.find("article")
        or soup.body
        or soup
    )


def _table_to_md(table) -> str:
    rows_md: list[str] = []
    for i, tr in enumerate(table.find_all("tr")):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["th", "td"])]
        row = "| " + " | ".join(cells) + " |"
        rows_md.append(row)
        if i == 0 and len(cells) > 0:
            sep = "| " + " | ".join(["---"] * len(cells)) + " |"
            rows_md.append(sep)
    return "\n".join(rows_md)


def _node_to_md(node, lines: list[str], base_url: str) -> None:
    """Recursively walk a BeautifulSoup node and append Markdown lines."""
    if isinstance(node, NavigableString):
        text = str(node).strip()
        if text:
            lines.append(text)
        return

    name = node.name
    if name is None:
        return

    if name == "h1":
        lines.append(f"\n# {node.get_text(' ', strip=True)}\n")
    elif name == "h2":
        lines.append(f"\n## {node.get_text(' ', strip=True)}\n")
    elif name == "h3":
        lines.append(f"\n### {node.get_text(' ', strip=True)}\n")
    elif name in ("h4", "h5", "h6"):
        lines.append(f"\n#### {node.get_text(' ', strip=True)}\n")
    elif name == "p":
        text = node.get_text(" ", strip=True)
        if text:
            lines.append(f"\n{text}\n")
    elif name in ("ul", "ol"):
        lines.append("")
        for li in node.find_all("li", recursive=False):
            lines.append(f"- {li.get_text(' ', strip=True)}")
        lines.append("")
    elif name == "li":
        lines.append(f"- {node.get_text(' ', strip=True)}")
    elif name == "a":
        href = node.get("href", "")
        link_text = node.get_text(strip=True)
        if href and not href.startswith("#") and link_text:
            full = urljoin(base_url, href)
            lines.append(f"[{link_text}]({full})")
        elif link_text:
            lines.append(link_text)
    elif name in ("strong", "b"):
        t = node.get_text(strip=True)
        if t:
            lines.append(f"**{t}**")
    elif name in ("em", "i"):
        t = node.get_text(strip=True)
        if t:
            lines.append(f"*{t}*")
    elif name == "table":
        lines.append("\n" + _table_to_md(node) + "\n")
    elif name == "br":
        lines.append("\n")
    elif name == "hr":
        lines.append("\n---\n")
    elif name == "blockquote":
        for child in node.children:
            sub: list[str] = []
            _node_to_md(child, sub, base_url)
            for s in sub:
                lines.append(f"> {s}")
    else:
        for child in node.children:
            _node_to_md(child, lines, base_url)


def html_to_markdown(html: str, base_url: str = "", page_title: str = "") -> str:
    soup = BeautifulSoup(html, "lxml")
    title = page_title or (soup.title.get_text(strip=True) if soup.title else "")
    _strip_junk(soup)
    main = _find_main(soup)

    lines: list[str] = []
    if title:
        lines.append(f"# {title}\n")

    for child in main.children:
        _node_to_md(child, lines, base_url)

    md = " ".join(lines)
    # Normalise whitespace
    md = re.sub(r"[ \t]+", " ", md)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


# ── URL helpers ───────────────────────────────────────────────────────────────

def is_allowed_url(url: str) -> bool:
    p = urlparse(url)
    return (
        p.scheme in ("http", "https")
        and ALLOWED_DOMAIN in p.netloc
        and not p.path.endswith((".pdf", ".doc", ".docx", ".xls", ".png", ".jpg", ".jpeg", ".gif", ".zip"))
    )


def is_time_sensitive(url: str) -> bool:
    return bool(_TS_RE.search(url.lower()))


def canonical(url: str) -> str:
    """Remove fragment and normalise trailing slash."""
    p = urlparse(url)
    clean = p._replace(fragment="", query="").geturl()
    return clean.rstrip("/") + "/"


def extract_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    seen: set[str] = set()
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        full = urljoin(base_url, a["href"])
        c = canonical(full)
        if is_allowed_url(c) and c not in seen:
            seen.add(c)
            links.append(c)
    return links


def url_to_filename(url: str) -> str:
    slug = re.sub(r"[^\w\-]", "_", url.replace("https://", "").replace("http://", ""))
    slug = re.sub(r"_+", "_", slug).strip("_")[:80]
    digest = hashlib.md5(url.encode()).hexdigest()[:8]
    return f"{slug}_{digest}.md"


# ── main crawler ──────────────────────────────────────────────────────────────

def crawl(max_pages: int = MAX_PAGES, output_dir: Path = OUTPUT_DIR) -> list[dict]:
    output_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    queue: list[str] = [canonical(u) for u in SEED_URLS]
    visited: set[str] = set()
    index: list[dict] = []

    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        if not is_allowed_url(url):
            continue
        if not can_fetch(url):
            log.info("robots.txt disallows %s — skipping", url)
            visited.add(url)
            continue

        visited.add(url)
        log.info("[%d/%d] Fetching %s", len(visited), max_pages, url)

        resp = None
        for attempt in range(1, 5):
            try:
                resp = session.get(url, timeout=15, allow_redirects=True)
                if resp.status_code == 429:
                    retry_after = int(resp.headers.get("Retry-After", 0))
                    backoff = retry_after if retry_after > 0 else min(30 * attempt, 120)
                    log.warning("429 rate-limited — waiting %ds (attempt %d)", backoff, attempt)
                    time.sleep(backoff)
                    resp = None
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                log.warning("Fetch failed %s (attempt %d): %s", url, attempt, exc)
                time.sleep(REQUEST_DELAY * attempt)
                resp = None

        if resp is None:
            log.warning("Giving up on %s after retries", url)
            time.sleep(REQUEST_DELAY)
            continue

        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            time.sleep(REQUEST_DELAY)
            continue

        html = resp.text
        md = html_to_markdown(html, base_url=url)
        if len(md.strip()) < 100:      # skip near-empty pages
            time.sleep(REQUEST_DELAY)
            continue

        filename = url_to_filename(url)
        (output_dir / filename).write_text(md, encoding="utf-8")

        sensitive = is_time_sensitive(url)
        entry = {
            "url": url,
            "filename": filename,
            "time_sensitive": sensitive,
            "scraped_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "content_hash": hashlib.md5(md.encode()).hexdigest(),
        }
        index.append(entry)

        new_links = [
            lnk for lnk in extract_links(html, url)
            if lnk not in visited and lnk not in queue
        ]
        queue.extend(new_links[:MAX_LINKS_PER_PAGE])

        time.sleep(REQUEST_DELAY)

    manifest_path = output_dir / "_index.json"
    manifest_path.write_text(json.dumps(index, indent=2), encoding="utf-8")

    n_sensitive = sum(1 for e in index if e["time_sensitive"])
    log.info(
        "Crawl complete — %d pages saved (%d time-sensitive). Manifest: %s",
        len(index), n_sensitive, manifest_path,
    )
    return index


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI'dvisor McGill web scraper")
    parser.add_argument(
        "--max-pages", type=int, default=MAX_PAGES,
        help=f"Maximum pages to crawl (default: {MAX_PAGES})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=OUTPUT_DIR,
        help="Where to write scraped files",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Scrape only seed URLs without following links (useful for testing)",
    )
    args = parser.parse_args()

    if args.quick:
        MAX_LINKS_PER_PAGE = 0
        crawl(max_pages=len(SEED_URLS), output_dir=args.output_dir)
    else:
        crawl(max_pages=args.max_pages, output_dir=args.output_dir)
