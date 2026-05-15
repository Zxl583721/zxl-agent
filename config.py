import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"

load_dotenv(ENV_PATH)


ZHIPUAI_API_KEY = os.getenv("ZHIPUAI_API_KEY", "")

