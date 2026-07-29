# Portfolio RAG Chatbot

A retrieval-augmented chatbot that answers questions about my background and projects. Retrieves relevant passages from my resume and supporting docs, then generates a grounded answer.

**Live:** `https://kartiks-chatbot.onrender.com/docs`

```
offline:  documents → chunk → embed → Postgres/pgvector
online:   question → embed → top-k search → prompt with context → answer
```

## Stack

FastAPI · Postgres + pgvector (Supabase) · Redis · Gemini (`gemini-embedding-001` @ 768d for retrieval, `gemini-3.1-flash-lite` for generation) · Logtail · Sentry · Render

One database for vectors and relational data. Hosted embeddings keep the container small enough for a 512MB instance.

## Endpoints

- `GET /health` — liveness check
- `POST /chat` — `{"question": "..."}` → `{"answer": "...", "sources": [...]}`

## Running locally

```bash
pip install -r requirements.txt
cp .env.example .env      # DATABASE_URL, REDIS_URL, GEMINI_API_KEY
mkdir data                # drop .txt / .md / .pdf here
python ingest.py
uvicorn main:app --reload
```

Optional: `ALLOWED_ORIGINS`, `SENTRY_DSN`, `LOGTAIL_SOURCE_TOKEN`, `LOGTAIL_HOST`. `init_db()` runs on startup and is idempotent.

## Design decisions

**Idempotent ingestion** — re-running `ingest.py` replaces a document's chunks by `source` instead of duplicating them, so updates are a re-run.

**Asymmetric embeddings** — docs use `RETRIEVAL_DOCUMENT`, queries use `RETRIEVAL_QUERY`. Same text, different vector, measurably better retrieval.

**768 dims, not the 3072 default** — less storage, faster search, negligible quality cost at this corpus size. `EMBEDDING_DIM` and `EMBEDDING_OUTPUT_DIM` must stay in sync; changing either means re-embedding everything.

**Non-blocking persistence** — a failed `chat_log` write is logged, not raised. The answer was already generated and paid for.

## Known limitations

Understood tradeoffs, not oversights.

- **ivfflat index is built pre-ingest** — centroids come from an empty table, so approximate search quality is poor. Fix: build after ingest with `lists ≈ sqrt(rows)`, or move to HNSW.
- **New Postgres connection per query** — handshake cost and pooler slots wasted. Fix: `psycopg_pool.ConnectionPool`.
- **Retrieval distance computed but unused** — thresholding it would let the service abstain instead of answering from weak context.
- **Rate limit is global, not per-identity** — a cost circuit-breaker, not abuse protection.
- **Embedding is one request per chunk** — sequential calls against a per-minute quota. Needs batching and backoff.
- **No retrieval eval yet** — chunk size, overlap, and top-k are untuned defaults. `chat_log` is the intended sampling frame for a golden set.

## Layout

```
main.py             app, routing, rate limiting
rag.py              chunk, embed, retrieve, generate
db.py               schema, vector search, persistence
ingest.py           reads ./data → ingest_text()
logging_config.py   root logger + Logtail
```

## Roadmap

Retrieval eval (recall@k, MRR) · hybrid search · streaming · abstention · connection pooling
