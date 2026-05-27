from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.models import User, KnowledgeBase


def seed_defaults(db: Session) -> None:
    settings = get_settings()

    user = db.get(User, settings.default_user_id)
    if user is None:
        user = User(
            id=settings.default_user_id,
            username="default_user",
            display_name="Default User",
            is_active=True,
        )
        db.add(user)

    knowledge_base = db.get(KnowledgeBase, settings.default_knowledge_base_id)
    if knowledge_base is None:
        knowledge_base = KnowledgeBase(
            id=settings.default_knowledge_base_id,
            user_id=settings.default_user_id,
            name="默认知识库",
            description="Local Chroma-backed knowledge base.",
        )
        db.add(knowledge_base)

    db.commit()

