"""
FastAPI wrapper around the RAG pipeline.

Endpoints:
  GET  /health   -> quick check that it's alive
  POST /chat     -> ask a question, get an answer  (this is what your FE calls)

Observability:
  - Structured logging (shows in Render/Docker logs): startup, requests, timings.
  - Sentry (optional): set SENTRY_DSN to get error tracking + alerts.

Start locally:   uvicorn main:app --reload
Start on Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis

load_dotenv()

# --- Logging setup: one place, applies to all modules (rag, db, main) ---
from logging_config import setup_logging
setup_logging()
logger = logging.getLogger("main")

# --- Sentry (optional error tracking). Only activates if SENTRY_DSN is set. ---
SENTRY_DSN = os.getenv("SENTRY_DSN")
if SENTRY_DSN:
    import sentry_sdk
    sentry_sdk.init(dsn=SENTRY_DSN, traces_sample_rate=0.2)
    logger.info("Sentry error tracking enabled")


from db import init_db,save_chat          # noqa: E402  (import after load_dotenv)
from rag import answer_question  # noqa: E402


# --- Redis (rate limiting). Standard client over TCP; same as AWS ElastiCache. ---
redis_client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)

RATE_LIMIT_MAX = 10        # max requests...
RATE_LIMIT_WINDOW = 3600  # ...per this many seconds (1 hour), global


def check_rate_limit():
    """Global cap: at most RATE_LIMIT_MAX requests per RATE_LIMIT_WINDOW.
    Redis INCR + EXPIRE: atomic counter that auto-resets after the window."""
    key = "ratelimit:global"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, RATE_LIMIT_WINDOW)
    if count > RATE_LIMIT_MAX:
        logger.warning("rate limit hit (count=%d)", count)
        raise HTTPException(status_code=429, detail="Too many requests. Try again later.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: initializing DB...")
    init_db()
    logger.info("startup: ready.")
    yield
    logger.info("shutdown.")


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

# --- CORS: only let your own frontend origins call this from a browser ---
origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat")
def chat(req: ChatRequest):
    check_rate_limit()
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    logger.info("chat request received")
    try:
        result = answer_question(req.question)
        logger.info("chat request answered (%d sources)", len(result["sources"]))
        
    except Exception:
        # log the full traceback; Sentry (if enabled) also captures it
        logger.exception("chat request failed")
        raise HTTPException(status_code=500, detail="Something went wrong.")    
    try:
        save_chat(req.question, result["answer"])
        logger.info("chat saved to database")
    except Exception as e:
        logger.error(f"failed to save chat to database: {e}")
        raise HTTPException(status_code=500, detail="Failed to save chat to database.")

    return result
