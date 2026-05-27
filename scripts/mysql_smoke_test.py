import json
import sys
from pathlib import Path
from uuid import uuid4

from sqlalchemy import select, text
from sqlalchemy.exc import OperationalError

BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.core.config import get_settings
from app.db.base import Base
from app.db.init_db import seed_defaults
from app.db.session import SessionLocal, engine
from app.models import ChatMessage, ChatSession, Document, DocumentStatus, KnowledgeBase, TaskRecord, TaskStatus, User


def main() -> None:
    settings = get_settings()
    if not settings.database_url.startswith("mysql+pymysql://"):
        raise SystemExit(f"当前不是 MySQL 连接，已停止：{mask_url(settings.database_url)}")

    print(f"Connecting to: {mask_url(settings.database_url)}")
    try:
        Base.metadata.create_all(bind=engine)

        with SessionLocal() as db:
            seed_defaults(db)

            suffix = uuid4().hex[:8]
            user = User(
                username=f"smoke_user_{suffix}",
                display_name="MySQL Smoke Test User",
                is_active=True,
            )
            db.add(user)
            db.flush()

            knowledge_base = KnowledgeBase(
                user_id=user.id,
                name=f"smoke_kb_{suffix}",
                description="Inserted by scripts/mysql_smoke_test.py",
            )
            db.add(knowledge_base)
            db.flush()

            document = Document(
                user_id=user.id,
                knowledge_base_id=knowledge_base.id,
                filename=f"smoke_document_{suffix}.txt",
                file_path=f"/tmp/smoke_document_{suffix}.txt",
                file_type="txt",
                file_size=128,
                file_hash=f"smoke_hash_{suffix}",
                status=DocumentStatus.COMPLETED.value,
                chunk_count=3,
            )
            db.add(document)
            db.flush()

            session = ChatSession(
                conversation_id=f"smoke_conversation_{suffix}",
                user_id=user.id,
                knowledge_base_id=knowledge_base.id,
                title="MySQL smoke test conversation",
            )
            db.add(session)
            db.flush()

            db.add_all(
                [
                    ChatMessage(
                        session_id=session.id,
                        user_id=user.id,
                        role="user",
                        content="证明 MySQL 已经接入了吗？",
                        mode="knowledge",
                        sources_json="[]",
                    ),
                    ChatMessage(
                        session_id=session.id,
                        user_id=user.id,
                        role="assistant",
                        content="是的，这条问答记录来自 MySQL smoke test。",
                        mode="knowledge",
                        sources_json=json.dumps([{"source_id": 1, "source": document.filename}], ensure_ascii=False),
                    ),
                ]
            )

            task = TaskRecord(
                task_id=f"smoke_task_{suffix}",
                task_type="mysql_smoke_test",
                status=TaskStatus.COMPLETED.value,
                user_id=user.id,
                knowledge_base_id=knowledge_base.id,
                document_id=document.id,
            )
            db.add(task)
            db.commit()

            rows = db.execute(
                text(
                    """
                    SELECT
                        u.id AS user_id,
                        u.username,
                        kb.id AS knowledge_base_id,
                        kb.name AS knowledge_base_name,
                        d.id AS document_id,
                        d.filename,
                        d.status AS document_status,
                        cs.conversation_id,
                        COUNT(cm.id) AS message_count,
                        tr.task_id,
                        tr.status AS task_status
                    FROM users u
                    JOIN knowledge_bases kb ON kb.user_id = u.id
                    JOIN documents d ON d.knowledge_base_id = kb.id
                    JOIN chat_sessions cs ON cs.knowledge_base_id = kb.id
                    JOIN chat_messages cm ON cm.session_id = cs.id
                    JOIN task_records tr ON tr.document_id = d.id
                    WHERE u.username = :username
                    GROUP BY
                        u.id, u.username, kb.id, kb.name, d.id, d.filename,
                        d.status, cs.conversation_id, tr.task_id, tr.status
                    """
                ),
                {"username": user.username},
            ).mappings().all()

            print("Inserted and queried MySQL rows:")
            for row in rows:
                print(dict(row))

            table_counts = {}
            for table_name in Base.metadata.tables:
                table_counts[table_name] = db.scalar(text(f"SELECT COUNT(*) FROM {table_name}"))
            print("Current table counts:")
            print(table_counts)
    except OperationalError as exc:
        raise SystemExit(
            "MySQL 连接失败。请确认 MySQL 服务已启动、数据库已创建、账号密码正确。\n"
            f"当前连接：{mask_url(settings.database_url)}\n"
            f"底层错误：{exc.orig}"
        ) from exc


def mask_url(url: str) -> str:
    if "@" not in url or "://" not in url:
        return url
    scheme, rest = url.split("://", 1)
    if ":" not in rest.split("@", 1)[0]:
        return url
    user = rest.split(":", 1)[0]
    host_part = rest.split("@", 1)[1]
    return f"{scheme}://{user}:***@{host_part}"


if __name__ == "__main__":
    main()
