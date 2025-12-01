"""FastAPI application."""

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator

from fastapi import FastAPI, status
from fastapi.middleware.cors import CORSMiddleware

from propertyrag import __version__
from propertyrag.api.routes import documents, projects, query
from propertyrag.api.schemas import HealthResponse
from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger, setup_logging
from propertyrag.db.session import engine

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan handler."""
    # Startup
    setup_logging()
    logger.info("application_starting", version=__version__)

    yield

    # Shutdown
    logger.info("application_stopping")
    await engine.dispose()


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="PropertyRAG",
        description="RAG system for real estate document analysis",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(projects.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")

    # Health check endpoint
    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["health"],
    )
    async def health_check() -> HealthResponse:
        """Check application health."""
        # TODO: Add actual database health check
        return HealthResponse(
            status="healthy",
            version=__version__,
            database="connected",
        )

    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        """Root endpoint."""
        return {
            "name": "PropertyRAG",
            "version": __version__,
            "docs": "/docs",
        }

    return app


# Create app instance
app = create_app()
