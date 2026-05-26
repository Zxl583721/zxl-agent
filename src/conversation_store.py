from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from uuid import uuid4


VALID_ROLES = {"user", "assistant"}


class ConversationStore:
    """SQLite-backed storage for browser chat sessions."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_or_create_conversation(self, conversation_id: str | None = None) -> str:
        conversation_id = _clean_conversation_id(conversation_id) or uuid4().hex
        now = _utc_now()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO conversations (id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET updated_at = excluded.updated_at
                """,
                (conversation_id, now, now),
            )

        return conversation_id

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        mode: str = "knowledge",
        sources: list[dict] | None = None,
    ) -> None:
        if role not in VALID_ROLES:
            raise ValueError(f"Unsupported message role: {role}")

        content = str(content).strip()
        if not content:
            return

        now = _utc_now()
        sources_json = json.dumps(sources or [], ensure_ascii=False)

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO messages (conversation_id, role, content, mode, sources_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, role, content, mode, sources_json, now),
            )
            conn.execute(
                "UPDATE conversations SET updated_at = ? WHERE id = ?",
                (now, conversation_id),
            )

    def recent_messages(self, conversation_id: str, limit: int) -> list[dict]:
        if limit <= 0:
            return []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (conversation_id, limit),
            ).fetchall()

        return [
            {"role": row["role"], "content": row["content"]}
            for row in reversed(rows)
        ]

    def messages(self, conversation_id: str) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content, mode, sources_json, created_at
                FROM messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [_message_from_row(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    mode TEXT NOT NULL DEFAULT 'knowledge',
                    sources_json TEXT NOT NULL DEFAULT '[]',
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_id
                ON messages(conversation_id, id)
                """
            )


def _message_from_row(row: sqlite3.Row) -> dict:
    try:
        sources = json.loads(row["sources_json"] or "[]")
    except json.JSONDecodeError:
        sources = []

    return {
        "role": row["role"],
        "content": row["content"],
        "mode": row["mode"],
        "sources": sources if isinstance(sources, list) else [],
        "created_at": row["created_at"],
    }


def _clean_conversation_id(conversation_id: str | None) -> str:
    conversation_id = str(conversation_id or "").strip()
    if not conversation_id:
        return ""
    if len(conversation_id) > 80:
        return ""
    if not all(char.isalnum() or char in {"-", "_"} for char in conversation_id):
        return ""
    return conversation_id


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
