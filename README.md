# Marty.AI — McGill AI Student Advisor

An AI-powered student advisor for McGill Students. Skip the hussle and bustle of navigating the McGill online maze. Dont stress about getting an advisor meeting. Instead, Ask Marty!

---

## Architecture

```
scraper/scraper.py          Crawls mcgill.ca → Markdown files
knowledge/*.md              Hand-curated: careers, fairs, clubs, campus culture
         ↓
rag/ingest.py               Chunks + embeds both → ChromaDB
         ↓
events/feed.py              Live public McGill events (iCal) → events_cache.json
         ↓
rag/router.py               Routes queries (vector store / live fetch / events feed)
         ↓
backend/main.py             FastAPI — /chat/stream SSE, /chat, /events
         ↓
frontend/index.html         Single-file chat UI + "This week at McGill" strip
```

## What Marty knows

| Area | Where it comes from |
|---|---|
| Registration, aid, wellness, international students | Scraped mcgill.ca pages |
| Career fairs (2026–27 calendar), CaPS, faculty career centres, Mentor Program | `knowledge/` (curated, each file has `source_url` + `verified` date) + scraped career pages |
| Clubs, SSMU, Activities Night, finance/consulting/tech clubs | `knowledge/clubs_and_involvement.md` |
| Traditions, places, slang (OAP, Gerts, Frosh, the Y, U0–U3…) | `knowledge/campus_culture.md` |
| What's on this week | `events/feed.py` live feed |

**What Marty can't see:** Minerva, myCourses and myFuture all sit behind McGill login.
CaPS moved its own events calendar into myFuture, so Marty points students there for
CaPS workshops, job postings and advising bookings.

### Updating the curated knowledge
Edit any file in `knowledge/`, bump its `verified:` date, then re-run `python rag/ingest.py`.
The career-fair table needs a refresh each September when the new year's dates go up.

---

## Quick Start (local, without Docker)

### 1. Install dependencies

```bash
cd marty.ai
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and fill in your ANTHROPIC_API_KEY and OPENAI_API_KEY
```

### 3. Scrape McGill (quick test — seed URLs only)

```bash
python scraper/scraper.py --quick
# Full 200-page crawl:  python scraper/scraper.py
```

### 4. Ingest into ChromaDB

```bash
python rag/ingest.py
```

### 5. Start the backend

```bash
python backend/main.py
# API available at http://localhost:8000
# Docs at         http://localhost:8000/docs
```

### 6. Open the frontend

Open `frontend/index.html` in a browser, or serve it:

```bash
python -m http.server 3000 --directory frontend
# Then visit http://localhost:3000
```

---

## Running each component standalone

### Scraper
```bash
# Quick test (seed URLs only, no link-following):
python scraper/scraper.py --quick

# Limited crawl for development:
python scraper/scraper.py --max-pages 20

# Full production crawl (200 pages):
python scraper/scraper.py
```

Output: `scraper/scraped_data/*.md` + `scraper/scraped_data/_index.json`

### RAG Ingestion
```bash
python rag/ingest.py
```

### Query Router (test a query directly)
```bash
python rag/router.py "What are the add/drop deadlines this semester?"
python rag/router.py "How do I apply for a bursary?"
```

### Live events feed
```bash
python events/feed.py                     # crawl public McGill event pages, print next 14 days
python events/feed.py --days 30 --q "career"
python events/feed.py --no-refresh        # just read the cache
```
The backend refreshes the feed in a background thread on startup and every
`EVENTS_REFRESH_HOURS` (default 6). Set `EVENTS_FEED=off` to disable. Event sources are
listed in `EVENT_SOURCES` at the top of `events/feed.py`; a source that breaks is skipped.

`GET /events?days=7&tag=career&q=fair` returns the cached events as JSON.

### Tests
```bash
pip install pytest httpx
python -m pytest -q tests
```

### Refresh Scheduler
```bash
# Dry run — see what has changed without writing anything:
python scripts/refresh.py

# Apply changes (re-crawl, re-embed changed pages):
python scripts/refresh.py --apply

# Weekly cron (2 AM every Sunday):
# 0 2 * * 0  cd /path/to/marty.ai && /path/to/.venv/bin/python scripts/refresh.py --apply
```

---

## Swapping to free embeddings (no OpenAI key required)

1. Uncomment `sentence-transformers` in `requirements.txt` and reinstall:
   ```bash
   pip install sentence-transformers
   ```

2. Set the environment variable before running ingest or the backend:
   ```bash
   export EMBED_BACKEND=local
   # or in .env:
   EMBED_BACKEND=local
   ```

3. Re-run `python rag/ingest.py` to rebuild the index with local embeddings.

> Note: `all-MiniLM-L6-v2` produces 384-dimensional embeddings vs 1536 for
> `text-embedding-3-small`. Quality is slightly lower but completely free.
> The ChromaDB collection must be rebuilt when switching embedding backends —
> you cannot mix dimensions in one collection.

---

## Running with Docker

```bash
cp .env.example .env
# Fill in your API keys in .env

docker compose up --build
```

Services:
- **Frontend** → http://localhost:8080
- **Backend API** → http://localhost:8000
- **ChromaDB** → http://localhost:8001

---

## Deployment (Railway / Render)

### Railway

1. Push this repo to GitHub.
2. Create a new Railway project, connect your repo.
3. Add a **ChromaDB** service from the Railway template (or use Railway volumes).
4. Add a **Python** service for the backend:
   - Build command: `pip install -r requirements.txt`
   - Start command: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
5. Set environment variables: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `FRONTEND_URL`.
6. Deploy `frontend/index.html` to any static host (Netlify, Vercel, Cloudflare Pages).
   Update `API_BASE` at the top of the `<script>` block to point to your Railway backend URL.

### Render

1. Create a **Web Service** for the backend from your repo.
   - Build: `pip install -r requirements.txt`
   - Start: `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`
2. Create a **Persistent Disk** and mount it at `/app/chroma_data`.
3. Set environment variables in the Render dashboard.
4. Deploy the frontend as a **Static Site** (root: `frontend/`).
   Set the `API_BASE` constant in `frontend/index.html` to your Render backend URL.

---

## Project structure

```
marty.ai/
├── scraper/
│   ├── scraper.py          Web crawler
│   └── scraped_data/       Markdown pages + _index.json
├── rag/
│   ├── ingest.py           Chunking, embedding, ChromaDB upsert
│   └── router.py           Hybrid query router
├── backend/
│   └── main.py             FastAPI app
├── frontend/
│   └── index.html          Single-file chat UI
├── scripts/
│   └── refresh.py          Weekly content refresh
├── logs/                   refresh.log + change_log.jsonl
├── chroma_data/            ChromaDB persistent storage (git-ignored)
├── docker-compose.yml
├── Dockerfile.backend
├── nginx.conf
├── requirements.txt
├── .env.example
└── README.md
```

---

## Emergency contacts (always shown by the advisor)

| Service | Contact |
|---|---|
| McGill Security (24/7) | 514-398-3000 |
| Student Wellness Hub | 514-398-6017 |
| Nightline (peer support, evenings) | 514-398-6246 |
