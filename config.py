import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1").strip().rstrip("/")

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1").strip().rstrip("/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "ollama")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen3:8b").strip()

EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", LLM_PROVIDER).strip().lower()
EMBEDDING_BASE_URL = os.getenv("EMBEDDING_BASE_URL", "http://localhost:11434").strip().rstrip("/")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "bge-m3").strip()
OLLAMA_TIMEOUT_SECONDS = int(os.getenv("OLLAMA_TIMEOUT_SECONDS", "120"))

# Retrieval reranking. Keep the lightweight implementation as the default and
# opt into the evaluated BGE cross-encoder through environment variables.
RERANKER_PROVIDER = os.getenv("RERANKER_PROVIDER", "lightweight").strip().lower()
RERANKER_FALLBACK = os.getenv("RERANKER_FALLBACK", "lightweight").strip().lower()
BGE_RERANKER_MODEL_PATH = os.getenv(
    "BGE_RERANKER_MODEL_PATH",
    str(BASE_DIR / "models" / "bge-reranker-v2-m3"),
).strip()
BGE_RERANKER_USE_FP16 = os.getenv("BGE_RERANKER_USE_FP16", "false").strip().lower() == "true"
BGE_RERANKER_BATCH_SIZE = int(os.getenv("BGE_RERANKER_BATCH_SIZE", "8"))
