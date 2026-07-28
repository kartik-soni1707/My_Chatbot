"""
FastAPI wrapper around the RAG pipeline.

Endpoints:
  GET  /health   -> quick check that it's alive
  POST /chat     -> ask a question, get an answer  (this is what your FE calls)

Ingestion is done offline by running `python ingest.py` locally (reads ./data),
so there's no public ingest endpoint — the corpus only changes when you run it.

Start locally:   uvicorn main:app --reload
Start on Render: uvicorn main:app --host 0.0.0.0 --port $PORT
"""
import os
from contextlib import asynccontextmanager
import redis
from requests import Request
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

from db import init_db          # noqa: E402  (import after load_dotenv)
from rag import answer_question  # noqa: E402

# --- Redis connection (rate limiting). Standard `redis` client over TCP. ---
# Same code you'd write against AWS ElastiCache; only the URL differs.
redis_client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)
 
RATE_LIMIT_MAX = 10        # max requests...
RATE_LIMIT_WINDOW = 3600  # ...per this many seconds (1 hour) 
 
 
def check_rate_limit(request: Request):
    """Allow at most RATE_LIMIT_MAX requests per IP per RATE_LIMIT_WINDOW.
    Uses Redis INCR + EXPIRE: atomic counter that auto-resets after the window.
    Runs at the endpoint (the 'gate'), before any expensive RAG work."""
    
    key = f"ratelimit:global"
    # INCR is atomic: first call returns 1, and we set the window expiry then.
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, RATE_LIMIT_WINDOW)
    if count > RATE_LIMIT_MAX:
        raise HTTPException(
            status_code=429,
            detail="Too many requests. Please try again later.",
        )
    
@asynccontextmanager
async def lifespan(app: FastAPI):
    # runs once on startup: make sure the table + extension exist
    init_db()
    yield


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
    check_rate_limit(req)
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question is empty")
    return answer_question(req.question)