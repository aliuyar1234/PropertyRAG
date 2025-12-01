"""RAG (Retrieval-Augmented Generation) query service."""

from dataclasses import dataclass
from uuid import UUID

from openai import AsyncOpenAI
from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger
from propertyrag.core.models import QueryRequest, QueryResponse, Source
from propertyrag.services.retriever import RetrievedChunk, Retriever

logger = get_logger(__name__)

# System prompt for RAG
RAG_SYSTEM_PROMPT = """Du bist ein Experte für Immobiliendokumente und beantwortest Fragen basierend auf den bereitgestellten Dokumentausschnitten.

Regeln:
1. Antworte NUR basierend auf den bereitgestellten Quellen
2. Wenn die Quellen keine Antwort enthalten, sage das ehrlich
3. Zitiere relevante Quellen mit [Quellenname, Seite X]
4. Antworte auf Deutsch
5. Sei präzise und konkret, vermeide Spekulationen
6. Bei Zahlen und Daten: gib sie exakt wie in den Quellen an"""

RAG_USER_PROMPT = """Beantworte die folgende Frage basierend auf den Dokumentausschnitten:

FRAGE: {question}

DOKUMENTAUSSCHNITTE:
{context}

Antworte präzise und zitiere die relevanten Quellen."""


class RAGError(Exception):
    """Error during RAG query processing."""

    pass


@dataclass
class RAGConfig:
    """Configuration for RAG queries."""

    top_k: int = 5
    include_context: bool = True
    context_chunks: int = 1
    min_score: float = 0.3
    max_tokens: int = 1000
    temperature: float = 0.1


class RAGService:
    """
    Service for RAG-based question answering.

    Combines retrieval and generation to answer questions
    about documents using relevant context.
    """

    def __init__(
        self,
        session: AsyncSession,
        retriever: Retriever | None = None,
        client: AsyncOpenAI | None = None,
        config: RAGConfig | None = None,
    ) -> None:
        """
        Initialize the RAG service.

        Args:
            session: Database session.
            retriever: Optional retriever. Creates default if not provided.
            client: Optional OpenAI client. Creates default if not provided.
            config: Optional RAG configuration.
        """
        settings = get_settings()

        self.session = session
        self.retriever = retriever or Retriever(session)
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model
        self.config = config or RAGConfig()

        logger.info(
            "rag_service_initialized",
            model=self.model,
            top_k=self.config.top_k,
        )

    async def query(self, request: QueryRequest) -> QueryResponse:
        """
        Answer a question using RAG.

        Args:
            request: Query request with question and optional filters.

        Returns:
            QueryResponse with answer and sources.

        Raises:
            RAGError: If query processing fails.
        """
        logger.info(
            "rag_query_started",
            question_length=len(request.question),
            project_id=str(request.project_id) if request.project_id else None,
        )

        try:
            # Retrieve relevant chunks
            if self.config.include_context:
                chunks = await self.retriever.retrieve_with_context(
                    query=request.question,
                    top_k=request.top_k or self.config.top_k,
                    project_id=request.project_id,
                    document_ids=request.document_ids,
                    context_chunks=self.config.context_chunks,
                )
            else:
                chunks = await self.retriever.retrieve(
                    query=request.question,
                    top_k=request.top_k or self.config.top_k,
                    project_id=request.project_id,
                    document_ids=request.document_ids,
                    min_score=self.config.min_score,
                )

            if not chunks:
                logger.warning("no_chunks_found", question=request.question[:100])
                return QueryResponse(
                    answer="Ich konnte keine relevanten Informationen in den Dokumenten finden.",
                    sources=[],
                    query=request.question,
                )

            # Build context from chunks
            context = self._build_context(chunks)

            # Generate answer
            answer = await self._generate_answer(request.question, context)

            # Build sources
            sources = self._build_sources(chunks)

            logger.info(
                "rag_query_completed",
                chunk_count=len(chunks),
                source_count=len(sources),
                answer_length=len(answer),
            )

            return QueryResponse(
                answer=answer,
                sources=sources,
                query=request.question,
            )

        except Exception as e:
            logger.error("rag_query_error", error=str(e))
            raise RAGError(f"Query failed: {e}") from e

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        """Build context string from retrieved chunks."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            page_info = f", Seite {chunk.page_number}" if chunk.page_number else ""
            header = f"[Quelle {i}: {chunk.filename}{page_info}]"
            context_parts.append(f"{header}\n{chunk.content}")

        return "\n\n---\n\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using the LLM."""
        user_prompt = RAG_USER_PROMPT.format(
            question=question,
            context=context,
        )

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": RAG_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
        )

        return response.choices[0].message.content.strip()

    def _build_sources(self, chunks: list[RetrievedChunk]) -> list[Source]:
        """Build deduplicated source list from chunks."""
        # Deduplicate by document and page
        seen: set[tuple[UUID, int | None]] = set()
        sources: list[Source] = []

        for chunk in chunks:
            key = (chunk.document_id, chunk.page_number)
            if key in seen:
                continue
            seen.add(key)

            sources.append(
                Source(
                    document_id=chunk.document_id,
                    filename=chunk.filename,
                    page_number=chunk.page_number,
                    chunk_content=chunk.content[:500],  # Truncate for response
                    score=chunk.score,
                )
            )

        # Sort by score descending
        sources.sort(key=lambda s: s.score, reverse=True)

        return sources

    async def query_simple(
        self,
        question: str,
        project_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
    ) -> str:
        """
        Simple query that returns just the answer text.

        Convenience method for quick queries.

        Args:
            question: The question to answer.
            project_id: Optional project filter.
            document_ids: Optional document filter.

        Returns:
            Answer text.
        """
        request = QueryRequest(
            question=question,
            project_id=project_id,
            document_ids=document_ids,
        )
        response = await self.query(request)
        return response.answer
