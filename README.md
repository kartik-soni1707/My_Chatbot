# Portfolio RAG Chatbot

A retrieval-augmented chatbot that answers questions about my background, experience, and projects. Visitors ask questions on my portfolio site; the service retrieves relevant passages from my resume and supporting documents, and generates a grounded answer.

Built as a production-shaped service rather than a notebook demo — connection handling, rate limiting, structured logging, error tracking, and persistent Q&A capture are all in place.

**Live API:** `https://kartiks-chatbot.onrender.com` · **Docs:** `/docs`

---

## How it works

```
OFFLINE (once, via ingest.py)
  documents → chunk → embed → store in Postgres/pgvector

ONLINE (per request, via POST /chat)
  question → embed → vector search (top-k) → prompt with context → answer
```

Retrieval and generation are deliberately separate concerns. Retrieval quality is measurable on its own and bounds everything downstream — if the right passage isn't retrieved, no amount of prompt tuning recovers it.

---

## Stack

| Layer | Choice | Why |
|---|---|---|
| API | FastAPI + Uvicorn | Async, typed request validation, free OpenAPI docs |
| Vector store | Postgres + pgvector (Supabase) | One database for vectors and relational data; no separate vector DB to operate |
| Embeddings | `gemini-embedding-001` @ 768d | Hosted — keeps the container small and RAM low |
| Generation | `gemini-3.1-flash-lite` | Extractive RAG doesn't need a frontier model; cost and latency matter more |
| Cache / limiter | Redis | Atomic `INCR` + `EXPIRE` counter |
| Logging | Logtail (root handler) | Ships logs from every module, survives container restarts |
| Errors | Sentry (optional) | Tracebacks with context; activates only if `SENTRY_DSN` is set |
| Hosting | Render | Deploys from Git, no infrastructure to manage |

---

## Endpoints

**`GET /health`** — liveness check. Returns `{"status": "ok"}`. Used by the uptime monitor to keep the free-tier instance warm.

**`POST /chat`**

```json
{ "question": "What has Kartik worked on with LLMs?" }
```

```json
{
  "answer": "...",
  "sources": ["resume.pdf", "projects.md"]
}
```

Returns `400` on an empty question, `429` when the rate limit is exceeded, `500` on generation failure.

---

## Running locally

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                              # then fill in the values
```

Ingest your documents, then start the server:

```bash
mkdir -p data          # drop .txt, .md, or .pdf files here
python ingest.py
uvicorn main:app --reload
```

Open `http://localhost:8000/docs` for an interactive client.

### Environment variables

| Variable | Required | Purpose |
|---|---|---|
| `DATABASE_URL` | yes | Postgres connection string (pgvector extension required) |
| `REDIS_URL` | yes | Redis connection string |
| `GEMINI_API_KEY` | yes | Embeddings and generation |
| `ALLOWED_ORIGINS` | no | Comma-separated CORS origins; defaults to `*` |
| `SENTRY_DSN` | no | Enables error tracking |
| `LOGTAIL_SOURCE_TOKEN` | no | Enables log shipping |
| `LOGTAIL_HOST` | no | Logtail ingest host |

`init_db()` runs on startup and is idempotent — it creates the extension, tables, and index if they don't exist.

---

## Design decisions

**Idempotent ingestion.** `ingest_text()` deletes all chunks for a given `source` before inserting. Re-running `ingest.py` replaces a document's chunks rather than duplicating them, which also makes document updates a re-run rather than a manual cleanup.

**Asymmetric embedding task types.** Documents are embedded with `RETRIEVAL_DOCUMENT`, queries with `RETRIEVAL_QUERY`. The model produces different vectors for the same text depending on whether it's being stored or searched with, and using the right one measurably improves retrieval.

**768 dimensions, not the 3072 default.** `gemini-embedding-001` supports truncation. Smaller vectors mean less storage and faster search, at a quality cost small enough not to matter at this corpus size. `EMBEDDING_DIM` in `db.py` and `EMBEDDING_OUTPUT_DIM` in `rag.py` must stay in sync — vectors from different models or dimensions aren't comparable, and changing either requires re-embedding the entire corpus.

**Grounding instruction in the system prompt.** The model is told to answer only from the provided context and to say it doesn't know otherwise. Without this, RAG systems confidently answer from irrelevant retrieved chunks.

**Global rate limit as a cost circuit-breaker.** A single Redis counter caps total requests per hour. This protects against runaway spend on the LLM API, not against individual abuse — see limitations.

**Non-blocking persistence.** A failed write to `chat_log` is logged but doesn't fail the request. The answer was already generated and paid for; bookkeeping shouldn't discard it.

---

## Known limitations

Documented deliberately — these are understood tradeoffs, not oversights.

**The ivfflat index is built before ingestion.** `init_db()` runs at startup against an empty table, so the index builds its cluster centroids from zero rows and silently returns approximate results of poor quality. The fix is to build the index after ingestion with `lists ≈ sqrt(row_count)`, or move to HNSW, which has better recall and no empty-table failure mode.

**A new Postgres connection is opened per query.** `get_conn()` does a fresh connect on every call, costing a TCP and TLS handshake per request and consuming pooler connection slots. A `psycopg_pool.ConnectionPool` created at startup is the correct fix.

**Retrieval distance is computed but unused.** `search_chunks` returns a cosine distance that's discarded. Thresholding on it would let the service abstain when nothing relevant is retrieved, rather than generating from weak context.

**Rate limiting is global, not per-identity.** One caller can exhaust the quota for everyone. The production pattern keys on user or IP with a global ceiling on top.

**Embedding is one HTTP request per chunk.** `embed_texts` loops over `_embed`, so ingesting a large document means many sequential API calls against a per-minute quota. Batching and retry with backoff would fix this.

**No retrieval evaluation yet.** There's no golden set, so there's no measurement of recall@k or MRR — which means chunk size, overlap, and top-k are currently untuned defaults rather than chosen values. Accumulated `chat_log` rows are the intended sampling frame for building one.

**Free-tier constraints.** The instance spins down after 15 minutes of inactivity and cold-starts in roughly a minute; an external uptime monitor pings `/health` to mitigate this. Free Postgres and Redis instances have their own expiry and pausing behaviour.

---

## Project layout

```
├── main.py             FastAPI app, routing, rate limiting, lifespan
├── rag.py              Chunking, embedding, retrieval, generation
├── db.py               Schema, connections, vector search, chat persistence
├── ingest.py           Reads ./data, extracts text, calls ingest_text()
├── logging_config.py   Root logger + Logtail handler
├── data/               Source documents (gitignored)
└── requirements.txt
```

---

## Roadmap

- Retrieval eval harness — golden set, recall@k, MRR
- Hybrid search (BM25 + vector) for exact-term queries
- Streaming responses
- Abstention on low-confidence retrieval
- Connection pooling
