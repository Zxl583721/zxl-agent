from sqlalchemy.orm import Session


def seed_defaults(db: Session) -> None:
    """Deprecated compatibility hook.

    Users and knowledge bases are now created through authenticated API flows,
    not through a global default tenant.
    """
    db.commit()
