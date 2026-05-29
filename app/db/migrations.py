from sqlalchemy import inspect, text

from app.db.session import engine


def apply_compat_migrations() -> list[str]:
    inspector = inspect(engine)
    if "users" not in inspector.get_table_names():
        return []

    columns = {column["name"] for column in inspector.get_columns("users")}
    statements = []
    if "password_hash" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN password_hash VARCHAR(255) NOT NULL DEFAULT ''")
    if "email" not in columns:
        statements.append("ALTER TABLE users ADD COLUMN email VARCHAR(255) NULL")

    if not statements:
        return []

    applied = []
    with engine.begin() as connection:
        for statement in statements:
            connection.execute(text(statement))
            applied.append(statement)
    return applied
