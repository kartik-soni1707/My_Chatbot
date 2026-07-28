"""
FastAPI wrapper around the RAG pipeline.
...
"""
import os
import logging
from contextlib import asynccontextmanager

from dotenv import load_dotenv
load_dotenv()                      # ← FIRST, so env vars exist for everything below

import redis
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from logging_config import setup_logging
setup_logging()
logger = logging.getLogger("main")

from db import init_db,save_chat          # noqa: E402
from rag import answer_question  # noqa: E402

redis_client = redis.from_url(os.environ["REDIS_URL"], decode_responses=True)

RATE_LIMIT_MAX = 100
RATE_LIMIT_WINDOW = 3600


def check_rate_limit():            # ← no Request param needed (global limit)
    key = "ratelimit:global"
    count = redis_client.incr(key)
    if count == 1:
        redis_client.expire(key, RATE_LIMIT_WINDOW)
    if count > RATE_LIMIT_MAX:
        logger.warning("rate limit hit (count=%d)", count)   # ← LOG
        raise HTTPException(status_code=429, detail="Too many requests. Please try again later.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("startup: initializing DB")        # ← LOG
    init_db()
    logger.info("startup: ready")                  # ← LOG
    yield
    logger.info("shutdown")                        # ← LOG


app = FastAPI(title="RAG Chatbot", lifespan=lifespan)

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
    logger.info("chat request: %s", req.question[:80])   # ← LOG the question
    try:
        result = answer_question(req.question)
        logger.info("chat answered (%d sources)", len(result["sources"]))  # ← LOG
    except Exception:
        logger.exception("chat request failed")          # ← LOG errors + traceback
        raise HTTPException(status_code=500, detail="Something went wrong.")
     # persist the Q&A — but don't let a save failure break the response
    try:
        save_chat(req.question, result["answer"])
    except Exception:
        logger.exception("failed to save chat")
    return result