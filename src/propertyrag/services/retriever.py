"""Document retrieval service."""

from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger
from propertyrag.db.repository import ChunkRepository, DocumentRepository
from propertyrag.services.embedder import Embedder

logger = get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A retrieved chunk with metadata."""

    chunk_id: UUID
    document_id: UUID
    filename: str
    content: str
    page_number: int | None
    score: float


class Retriever:
    """
    Service for retrieving relevant document chunks.

    Uses cosine similarity search on embeddings stored in PostgreSQL with pgvector.
    """

    def __init__(
        self,
        session: AsyncSession,
        embedder: Embedder | None = None,
    ) -> None:
        """
        Initialize the retriever.

        Args:
            session: Database session.
            embedder: Optional embedder for query embedding. Creates default if not provided.
        """
        self.session = session
        self.embedder = embedder or Embedder()
        self.chunk_repo = ChunkRepository(session)
        self.doc_repo = DocumentRepository(session)

        settings = get_settings()
        self.default_top_k = settings.retrieval_top_k

        logger.info("retriever_initialized", default_top_k=self.default_top_k)

    async def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        project_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks for a query.

        Args:
            query: Natural language query.
            top_k: Number of chunks to retrieve. Defaults to settings value.
            project_id: Optional project to filter by.
            document_ids: Optional list of document IDs to filter by.
            min_score: Minimum similarity score (0-1). Default 0.

        Returns:
            List of retrieved chunks sorted by relevance.
        """
        top_k = top_k or self.default_top_k

        logger.info(
            "retrieving_chunks",
            query_length=len(query),
            top_k=top_k,
            project_id=str(project_id) if project_id else None,
            document_count=len(document_ids) if document_ids else None,
        )

        # Generate query embedding
        query_embedding = await self.embedder.embed_text(query)

        # Search for similar chunks
        results = await self.chunk_repo.search_similar(
            embedding=query_embedding,
            top_k=top_k,
            project_id=project_id,
            document_ids=document_ids,
        )

        # Convert to RetrievedChunk with document info
        retrieved_chunks: list[RetrievedChunk] = []

        # Cache document lookups
        doc_cache: dict[UUID, str] = {}

        for chunk, score in results:
            # Filter by minimum score
            if score < min_score:
                continue

            # Get document filename (cached)
            if chunk.document_id not in doc_cache:
                doc = await self.doc_repo.get_by_id(chunk.document_id)
                doc_cache[chunk.document_id] = doc.filename if doc else "Unknown"

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=doc_cache[chunk.document_id],
                    content=chunk.content,
                    page_number=chunk.page_number,
                    score=score,
                )
            )

        logger.info(
            "chunks_retrieved",
            count=len(retrieved_chunks),
            top_score=retrieved_chunks[0].score if retrieved_chunks else 0,
        )

        return retrieved_chunks

    async def retrieve_with_context(
        self,
        query: str,
        top_k: int | None = None,
        project_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
        context_chunks: int = 1,
    ) -> list[RetrievedChunk]:
        """
        Retrieve chunks with surrounding context.

        Fetches additional chunks before and after each matched chunk
        for better context in answers.

        Args:
            query: Natural language query.
            top_k: Number of primary chunks to retrieve.
            project_id: Optional project to filter by.
            document_ids: Optional list of document IDs to filter by.
            context_chunks: Number of chunks before/after to include.

        Returns:
            List of retrieved chunks with context, deduplicated.
        """
        # Get primary chunks
        primary_chunks = await self.retrieve(
            query=query,
            top_k=top_k,
            project_id=project_id,
            document_ids=document_ids,
        )

        if not primary_chunks or context_chunks == 0:
            return primary_chunks

        # Gather all chunks including context
        all_chunks: dict[UUID, RetrievedChunk] = {}

        for chunk in primary_chunks:
            all_chunks[chunk.chunk_id] = chunk

            # Get surrounding chunks from same document
            doc_chunks = await self.chunk_repo.get_by_document(chunk.document_id)

            # Find the index of our chunk
            chunk_indices = {c.id: i for i, c in enumerate(doc_chunks)}
            current_idx = chunk_indices.get(chunk.chunk_id)

            if current_idx is not None:
                # Get context chunks
                start_idx = max(0, current_idx - context_chunks)
                end_idx = min(len(doc_chunks), current_idx + context_chunks + 1)

                for idx in range(start_idx, end_idx):
                    ctx_chunk = doc_chunks[idx]
                    if ctx_chunk.id not in all_chunks:
                        # Context chunks get a slightly lower score
                        context_score = chunk.score * 0.9
                        all_chunks[ctx_chunk.id] = RetrievedChunk(
                            chunk_id=ctx_chunk.id,
                            document_id=ctx_chunk.document_id,
                            filename=chunk.filename,
                            content=ctx_chunk.content,
                            page_number=ctx_chunk.page_number,
                            score=context_score,
                        )

        # Sort by document and chunk order for coherent reading
        result = sorted(
            all_chunks.values(),
            key=lambda c: (c.document_id, c.page_number or 0, c.score),
            reverse=False,
        )

        logger.info(
            "chunks_with_context_retrieved",
            primary_count=len(primary_chunks),
            total_count=len(result),
        )

        return result
