import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.logging import get_logger
from app.db.base import Base
from app.db.migrations import apply_compat_migrations
from app.db.session import engine

# Import models so SQLAlchemy registers table metadata before create_all.
import app.models  # noqa: F401


logger = get_logger(__name__)


def main() -> None:
    Base.metadata.create_all(bind=engine)
    for statement in apply_compat_migrations():
        logger.info("Applied compatibility migration: %s", statement)
    logger.info("Database tables are ready.")


if __name__ == "__main__":
    main()
