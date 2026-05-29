from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


class Settings:
    """Central application settings for the FastAPI service."""

    app_name = "Enterprise Knowledge Base RAG Agent"
    api_prefix = "/api"
    data_dir = BASE_DIR / "data"
    vector_store_dir = BASE_DIR / "vector_store"
    max_upload_bytes = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    log_level = os.getenv("LOG_LEVEL", "INFO")
    jwt_secret_key = os.getenv("JWT_SECRET_KEY", "change_me_to_a_32_plus_char_random_secret")
    jwt_algorithm = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_access_token_expire_minutes = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "1440"))

    mysql_host = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user = os.getenv("MYSQL_USER", "zxl_agent")
    mysql_password = os.getenv("MYSQL_PASSWORD", "change_me_strong_password")
    mysql_database = os.getenv("MYSQL_DATABASE", "zxl_agent")
    database_echo = os.getenv("DATABASE_ECHO", "false").strip().lower() == "true"
    redis_host = os.getenv("REDIS_HOST", "127.0.0.1")
    redis_port = int(os.getenv("REDIS_PORT", "6379"))
    redis_db = int(os.getenv("REDIS_DB", "0"))
    redis_password = os.getenv("REDIS_PASSWORD") or None
    redis_socket_timeout = float(os.getenv("REDIS_SOCKET_TIMEOUT", "1.0"))
    chat_rate_limit_per_minute = int(os.getenv("CHAT_RATE_LIMIT_PER_MINUTE", "10"))
    chat_cache_ttl_seconds = int(os.getenv("CHAT_CACHE_TTL_SECONDS", "300"))
    task_status_cache_ttl_seconds = int(os.getenv("TASK_STATUS_CACHE_TTL_SECONDS", "86400"))
    celery_task_serializer = os.getenv("CELERY_TASK_SERIALIZER", "json")
    celery_result_serializer = os.getenv("CELERY_RESULT_SERIALIZER", "json")
    celery_accept_content = os.getenv("CELERY_ACCEPT_CONTENT", "json")

    @property
    def database_url(self) -> str:
        explicit_url = os.getenv("DATABASE_URL", "").strip()
        if explicit_url:
            return explicit_url
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )

    @property
    def redis_url(self) -> str:
        auth = f":{self.redis_password}@" if self.redis_password else ""
        return f"redis://{auth}{self.redis_host}:{self.redis_port}/{self.redis_db}"

    @property
    def celery_broker_url(self) -> str:
        return os.getenv("CELERY_BROKER_URL", self.redis_url)

    @property
    def celery_result_backend(self) -> str:
        return os.getenv("CELERY_RESULT_BACKEND", self.redis_url)


@lru_cache
def get_settings() -> Settings:
    return Settings()
