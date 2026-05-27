from fastapi import FastAPI

from app.api.routes import router
from app.core.config import get_settings


def create_app() -> FastAPI:
    settings = get_settings()
    application = FastAPI(
        title=settings.app_name,
        description="FastAPI backend for an enterprise knowledge base RAG Agent.",
        version="0.1.0",
    )
    application.include_router(router)
    return application


app = create_app()

