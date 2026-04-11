#!/usr/bin/env python3
"""
AI'dvisor McGill Advisor — Weekly Refresh Scheduler
Re-crawls only time-sensitive pages (deadlines, fees, registration) and
re-embeds chunks whose content has changed since the last scrape.

Designed to run weekly via cron:
    0 2 * * 0  cd /path/to/minerva && python scripts/refresh.py

Usage:
    python scripts/refresh.py               # dry-run preview
    python scripts/refresh.py --apply       # actually re-ingest changed pages
    python scripts/refresh.py --all         # re-ingest all pages, not just time-sensitive
"""
import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

import requests
from bs4 import BeautifulSoup

from rag.ingest import ingest_file, get_collection

SCRAPED_DATA_DIR = Path(__file__).parent.parent / "scraper" / "scraped_data"
LOG_DIR = Path(__file__).parent.parent / "logs"
REQUEST_DELAY = 2.0
USER_AGENT = "AIdvisorBot/1.0 (educational)"

LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_DIR / "refresh.log", encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)

# ── helpers ───────────────────────────────────────────────────────────────────

# Reuse the same conversion logic as the scraper (import lazily)
_html_to_markdown = None


def get_converter():
    global _html_to_markdown
    if _html_to_markdown is None:
        from scraper.scraper import html_to_markdown
        _html_to_markdown = html_to_markdown
    return _html_to_markdown


def fetch_page(url: str) -> tuple[str, str]:
    """
    Fetch *url* and return (html, markdown).
    Raises requests.RequestException on failure.
    """
    resp = requests.get(
        url,
        timeout=15,
        headers={"User-Agent": USER_AGENT},
        allow_redirects=True,
    )
    resp.raise_for_status()
    html = resp.text
    converter = get_converter()
    md = converter(html, base_url=url)
    return html, md


def content_hash(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ── main refresh ──────────────────────────────────────────────────────────────

def run_refresh(apply: bool = False, all_pages: bool = False) -> None:
    manifest_path = SCRAPED_DATA_DIR / "_index.json"
    if not manifest_path.exists():
        log.error("Manifest not found at %s. Run the scraper first.", manifest_path)
        sys.exit(1)

    index: list[dict] = json.loads(manifest_path.read_text(encoding="utf-8"))

    if all_pages:
        candidates = index
        log.info("Mode: refreshing ALL %d pages", len(candidates))
    else:
        candidates = [e for e in index if e.get("time_sensitive")]
        log.info("Mode: refreshing %d time-sensitive pages", len(candidates))

    if not candidates:
        log.info("No pages to refresh. Exiting.")
        return

    changed: list[dict] = []
    unchanged: list[str] = []
    failed: list[str] = []
    change_log: list[dict] = []

    for entry in candidates:
        url = entry["url"]
        filename = entry["filename"]
        old_hash = entry.get("content_hash", "")
        filepath = SCRAPED_DATA_DIR / filename

        log.info("Checking %s", url)
        try:
            _, new_md = fetch_page(url)
        except requests.RequestException as exc:
            log.warning("  Fetch failed: %s", exc)
            failed.append(url)
            time.sleep(REQUEST_DELAY)
            continue

        new_hash = content_hash(new_md)

        if new_hash == old_hash:
            log.info("  Unchanged")
            unchanged.append(url)
        else:
            log.info("  CHANGED (hash %s → %s)", old_hash[:8], new_hash[:8])
            changed.append({"entry": entry, "new_md": new_md, "new_hash": new_hash})
            change_log.append(
                {
                    "url": url,
                    "filename": filename,
                    "old_hash": old_hash,
                    "new_hash": new_hash,
                    "checked_at": now_iso(),
                }
            )

        time.sleep(REQUEST_DELAY)

    log.info(
        "\nSummary — changed: %d  unchanged: %d  failed: %d",
        len(changed), len(unchanged), len(failed),
    )

    if not changed:
        log.info("Nothing to update. Vector store is current.")
        _write_change_log(change_log)
        return

    if not apply:
        log.info(
            "\nDRY RUN — re-run with --apply to write changes to disk and re-ingest.\n"
            "Changed pages:"
        )
        for item in changed:
            log.info("  %s", item["entry"]["url"])
        _write_change_log(change_log)
        return

    # ── apply changes ──────────────────────────────────────────────────────────
    updated_index = {e["url"]: e for e in index}

    for item in changed:
        entry = item["entry"]
        new_md = item["new_md"]
        new_hash = item["new_hash"]
        filepath = SCRAPED_DATA_DIR / entry["filename"]

        # 1. Write updated Markdown to disk
        filepath.write_text(new_md, encoding="utf-8")
        log.info("  Wrote %s", filepath.name)

        # 2. Re-ingest into ChromaDB (upsert replaces by chunk ID)
        n_chunks = ingest_file(filepath, entry["url"])
        log.info("  Re-ingested %d chunks for %s", n_chunks, entry["url"])

        # 3. Update manifest entry
        updated_index[entry["url"]]["content_hash"] = new_hash
        updated_index[entry["url"]]["scraped_at"] = now_iso()

    # 4. Write updated manifest
    manifest_path.write_text(
        json.dumps(list(updated_index.values()), indent=2),
        encoding="utf-8",
    )
    log.info("Manifest updated.")

    _write_change_log(change_log)
    log.info("Refresh complete. %d page(s) updated.", len(changed))


def _write_change_log(entries: list[dict]) -> None:
    log_path = LOG_DIR / "change_log.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")
    if entries:
        log.info("Change log appended to %s", log_path)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI'dvisor weekly content refresh")
    parser.add_argument(
        "--apply", action="store_true",
        help="Write changes to disk and re-ingest into ChromaDB (default: dry-run preview)",
    )
    parser.add_argument(
        "--all", dest="all_pages", action="store_true",
        help="Refresh all pages in the manifest, not just time-sensitive ones",
    )
    args = parser.parse_args()

    run_refresh(apply=args.apply, all_pages=args.all_pages)
