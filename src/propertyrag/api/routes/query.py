"""Query API routes."""

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.api.dependencies import get_db, get_rag_service
from propertyrag.api.schemas import QueryRequest, QueryResponse, SourceResponse
from propertyrag.core.models import QueryRequest as CoreQueryRequest
from propertyrag.services.rag import RAGError, RAGService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
async def query_documents(
    request: QueryRequest,
    session: AsyncSession = Depends(get_db),
) -> QueryResponse:
    """
    Query documents using natural language.

    Uses RAG (Retrieval-Augmented Generation) to:
    1. Find relevant document chunks
    2. Generate an answer based on the context
    3. Return the answer with source references

    Args:
        request: Query request with question and optional filters.

    Returns:
        Answer with source references.
    """
    rag_service = await get_rag_service(session)

    try:
        # Convert to core model
        core_request = CoreQueryRequest(
            question=request.question,
            project_id=request.project_id,
            document_ids=request.document_ids,
            top_k=request.top_k,
        )

        result = await rag_service.query(core_request)

        # Convert sources to response format
        sources = [
            SourceResponse(
                document_id=source.document_id,
                filename=source.filename,
                page_number=source.page_number,
                excerpt=source.chunk_content,
                score=source.score,
            )
            for source in result.sources
        ]

        return QueryResponse(
            answer=result.answer,
            sources=sources,
            query=result.query,
        )

    except RAGError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        )
