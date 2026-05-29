"""Deprecated Flask entrypoint."""

import logging


logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    logger.info("Flask web_app.py has been deprecated.")
    logger.info("Start the FastAPI app instead:")
    logger.info("  uvicorn app.main:app --host 0.0.0.0 --port 8000")


if __name__ == "__main__":
    main()
