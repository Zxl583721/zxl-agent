from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import BASE_DIR, get_settings
from app.core.logging import configure_logging, get_logger
from app.db.base import Base
from app.db.migrations import apply_compat_migrations
from app.db.session import engine
from app.services.rag_service import get_rag_service
from src.retriever import shutdown_retrieval_executor


logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    reranker_selection = get_rag_service().initialize_reranker()
    if reranker_selection.setup_error:
        logger.error(reranker_selection.setup_error)
    logger.info(
        "RAG reranker ready: requested=%s active=%s",
        reranker_selection.requested_provider,
        reranker_selection.active_provider,
    )
    try:
        Base.metadata.create_all(bind=engine)
        apply_compat_migrations()
    except Exception as exc:
        logger.warning("Database table initialization skipped: %s", exc.__class__.__name__)
    try:
        yield
    finally:
        shutdown_retrieval_executor()


def create_app() -> FastAPI:
    configure_logging()
    settings = get_settings()
    application = FastAPI(
        title="多用户知识库 RAG Agent",
        description="FastAPI 多用户 RAG/Agent 后端，支持 JWT 鉴权、知识库隔离、异步索引和健康检查。",
        version="1.0.0",
        contact={"name": settings.app_name},
        lifespan=lifespan,
    )

    static_dir = BASE_DIR / "static"
    if static_dir.exists():
        application.mount("/static", StaticFiles(directory=static_dir), name="static")

    @application.get("/", response_class=FileResponse, include_in_schema=False)
    def home() -> FileResponse:
        return FileResponse(Path(BASE_DIR) / "templates" / "index.html")

    application.include_router(router)
    return application


app = create_app()
