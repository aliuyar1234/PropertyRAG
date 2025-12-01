"""FastAPI dependencies."""

from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.db.session import async_session_maker
from propertyrag.services.ingestion import IngestionPipeline
from propertyrag.services.rag import RAGService


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


async def get_ingestion_pipeline(
    session: AsyncSession,
) -> IngestionPipeline:
    """Get an ingestion pipeline instance."""
    return IngestionPipeline(session)


async def get_rag_service(
    session: AsyncSession,
) -> RAGService:
    """Get a RAG service instance."""
    return RAGService(session)
