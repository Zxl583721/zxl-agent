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
    conversation_db_path = BASE_DIR / "conversation_store.sqlite3"
    default_user_id = int(os.getenv("DEFAULT_USER_ID", "1"))
    default_knowledge_base_id = int(os.getenv("DEFAULT_KNOWLEDGE_BASE_ID", "1"))

    mysql_host = os.getenv("MYSQL_HOST", "127.0.0.1")
    mysql_port = int(os.getenv("MYSQL_PORT", "3306"))
    mysql_user = os.getenv("MYSQL_USER", "zxl_agent")
    mysql_password = os.getenv("MYSQL_PASSWORD", "zxl_agent")
    mysql_database = os.getenv("MYSQL_DATABASE", "zxl_agent")
    database_echo = os.getenv("DATABASE_ECHO", "false").strip().lower() == "true"

    @property
    def database_url(self) -> str:
        explicit_url = os.getenv("DATABASE_URL", "").strip()
        if explicit_url:
            return explicit_url
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}?charset=utf8mb4"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
