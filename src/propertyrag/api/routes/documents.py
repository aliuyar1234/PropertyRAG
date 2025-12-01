"""Document API routes."""

from uuid import UUID

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.api.dependencies import get_db, get_ingestion_pipeline
from propertyrag.api.schemas import (
    DocumentListResponse,
    DocumentResponse,
    DocumentUploadResponse,
    ExtractedDataResponse,
)
from propertyrag.core.models import DocumentType, ProcessingStatus
from propertyrag.db.repository import DocumentRepository, ExtractedDataRepository
from propertyrag.services.ingestion import IngestionError, IngestionPipeline

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document(
    file: UploadFile = File(...),
    document_type: DocumentType = Form(default=DocumentType.UNKNOWN),
    project_id: UUID | None = Form(default=None),
    auto_extract: bool = Form(default=True),
    session: AsyncSession = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload a PDF document for processing.

    The document will be:
    1. Parsed to extract text
    2. Classified (if type is UNKNOWN)
    3. Chunked and embedded
    4. Optionally: structured data extracted

    Args:
        file: PDF file to upload.
        document_type: Type of document. If UNKNOWN, will be auto-classified.
        project_id: Optional project to associate with.
        auto_extract: Whether to extract structured data automatically.

    Returns:
        Upload response with document ID and status.
    """
    # Validate file type
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are supported",
        )

    # Read file content
    content = await file.read()
    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file",
        )

    # Process document
    pipeline = await get_ingestion_pipeline(session)

    try:
        document_id = await pipeline.ingest_bytes(
            content=content,
            filename=file.filename,
            document_type=document_type,
            project_id=project_id,
            auto_extract=auto_extract,
        )

        return DocumentUploadResponse(
            id=document_id,
            filename=file.filename,
            status=ProcessingStatus.COMPLETED,
            message="Document processed successfully",
        )

    except IngestionError as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        )


@router.get("", response_model=DocumentListResponse)
async def list_documents(
    project_id: UUID | None = None,
    session: AsyncSession = Depends(get_db),
) -> DocumentListResponse:
    """
    List all documents, optionally filtered by project.

    Args:
        project_id: Optional project ID to filter by.

    Returns:
        List of documents.
    """
    repo = DocumentRepository(session)

    if project_id:
        documents = await repo.get_by_project(project_id)
    else:
        documents = await repo.get_all()

    return DocumentListResponse(
        documents=[DocumentResponse.model_validate(doc) for doc in documents],
        total=len(documents),
    )


@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> DocumentResponse:
    """
    Get a document by ID.

    Args:
        document_id: Document ID.

    Returns:
        Document details.
    """
    repo = DocumentRepository(session)
    document = await repo.get_by_id(document_id)

    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )

    return DocumentResponse.model_validate(document)


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> None:
    """
    Delete a document and all its data.

    Args:
        document_id: Document ID.
    """
    repo = DocumentRepository(session)
    deleted = await repo.delete(document_id)

    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found",
        )


@router.get("/{document_id}/extracted", response_model=ExtractedDataResponse)
async def get_extracted_data(
    document_id: UUID,
    session: AsyncSession = Depends(get_db),
) -> ExtractedDataResponse:
    """
    Get extracted structured data for a document.

    Args:
        document_id: Document ID.

    Returns:
        Extracted data.
    """
    repo = ExtractedDataRepository(session)
    extracted = await repo.get_by_document(document_id)

    if not extracted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No extracted data found for this document",
        )

    return ExtractedDataResponse.model_validate(extracted)


@router.post("/{document_id}/extract", response_model=ExtractedDataResponse)
async def extract_document_data(
    document_id: UUID,
    force: bool = False,
    session: AsyncSession = Depends(get_db),
) -> ExtractedDataResponse:
    """
    Extract or re-extract structured data from a document.

    Args:
        document_id: Document ID.
        force: If True, re-extract even if data exists.

    Returns:
        Extracted data.
    """
    pipeline = await get_ingestion_pipeline(session)
    data = await pipeline.extract_document(document_id, force=force)

    if data is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Extraction failed or document type unknown",
        )

    # Get the full extracted data record
    repo = ExtractedDataRepository(session)
    extracted = await repo.get_by_document(document_id)

    return ExtractedDataResponse.model_validate(extracted)
