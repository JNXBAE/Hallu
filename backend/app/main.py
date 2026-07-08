"""
HalluZero — FastAPI entry point
"""
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from config.settings import get_settings

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: ensure data directories exist
    for path in ["./data/documents", "./data/chroma_db", "./data/uploads"]:
        os.makedirs(path, exist_ok=True)
    print(f"HalluZero API starting — LLM: {settings.ollama_llm_model}")
    yield
    print("HalluZero API shutting down")


app = FastAPI(
    title="HalluZero API",
    description="Anti-hallucination LLM stack with RAG + dual verifier + RLHF",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/")
async def root():
    return {"name": "HalluZero", "version": "1.0.0", "status": "running"}
