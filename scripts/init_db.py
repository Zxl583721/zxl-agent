import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.db.base import Base
from app.db.init_db import seed_defaults
from app.db.session import SessionLocal, engine

# Import models so SQLAlchemy registers table metadata before create_all.
import app.models  # noqa: F401


def main() -> None:
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db:
        seed_defaults(db)
    print("Database tables created and default records seeded.")


if __name__ == "__main__":
    main()
