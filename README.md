# Minimal RAG Chatbot (FastAPI + pgvector)

A small, deployable RAG skeleton. Preload your docs once, then answer questions
against them. Designed to sit behind your existing frontend and Postgres DB.

## What each file does
- `rag.py` — the RAG pipeline (chunk → embed → store → retrieve → generate). **This is the part to focus on.**
- `db.py` — Postgres + pgvector: stores your pre-computed embeddings.
- `main.py` — thin FastAPI wrapper: `/chat`, `/ingest`, `/health`. CORS wired in.
- `requirements.txt`, `.env.example` — deps and config template.

## Architecture (the "preload then use" split)
- **Offline (once, by you):** ingest docs → chunk → embed → store vectors. Not spammy, not exposed.
- **Online (per query):** embed the question → retrieve top chunks → send to the LLM → return the answer. The only public surface.

## Run locally
1. `python -m venv venv && source venv/bin/activate`
2. `pip install -r requirements.txt`
3. `cp .env.example .env` and fill in your keys + DATABASE_URL
4. `uvicorn main:app --reload`
5. Open http://localhost:8000/docs to try the endpoints interactively.

## Ingest a document (offline preload)
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"text": "your document text here", "source": "my-doc"}'
```

## Ask a question (what your frontend calls)
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "what is X?"}'
```

## Deploy to Render
1. Push this folder to a GitHub repo.
2. Render → New → **Web Service** → connect the repo.
3. **Build command:** `pip install -r requirements.txt`
4. **Start command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables (from your `.env`): `OPENAI_API_KEY`, `DATABASE_URL`, `ALLOWED_ORIGINS`, and optionally `ADMIN_TOKEN`.
6. Deploy → you get `https://your-app.onrender.com`. Your FE calls `POST /chat` there.

Note: Render's free tier sleeps after ~15 min idle (first request then has a ~40s cold start). The ~$7/mo tier stays warm.

## The one non-code safety step
Set a **hard spend cap** on your LLM provider dashboard. Your `/chat` endpoint is public and each call costs money on your key — the cap makes worst-case abuse harmless.

## Where to tune quality (in `rag.py`)
- **Chunking** (`chunk_text`): size + overlap. Biggest lever on quality.
- **top_k** in `answer_question`: how many chunks to retrieve.
- **The system prompt**: how strictly to stick to retrieved context.
- **Models**: `EMBEDDING_MODEL` and `CHAT_MODEL` — swap for cheaper/better as needed.
