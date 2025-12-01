"""API request/response schemas."""

from datetime import datetime
from decimal import Decimal
from uuid import UUID

from pydantic import BaseModel, Field

from propertyrag.core.models import DocumentType, ProcessingStatus


# ============================================================================
# Project Schemas
# ============================================================================


class ProjectCreate(BaseModel):
    """Request to create a project."""

    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None


class ProjectUpdate(BaseModel):
    """Request to update a project."""

    name: str | None = Field(None, min_length=1, max_length=255)
    description: str | None = None


class ProjectResponse(BaseModel):
    """Project response."""

    id: UUID
    name: str
    description: str | None
    document_count: int
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Document Schemas
# ============================================================================


class DocumentResponse(BaseModel):
    """Document response."""

    id: UUID
    filename: str
    document_type: DocumentType
    status: ProcessingStatus
    page_count: int | None
    project_id: UUID | None
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class DocumentListResponse(BaseModel):
    """List of documents response."""

    documents: list[DocumentResponse]
    total: int


class DocumentUploadResponse(BaseModel):
    """Response after document upload."""

    id: UUID
    filename: str
    status: ProcessingStatus
    message: str


# ============================================================================
# Extraction Schemas
# ============================================================================


class ExtractedDataResponse(BaseModel):
    """Extracted data response."""

    document_id: UUID
    document_type: DocumentType
    data: dict
    extraction_confidence: float | None
    extracted_at: datetime

    class Config:
        from_attributes = True


# ============================================================================
# Query Schemas
# ============================================================================


class QueryRequest(BaseModel):
    """RAG query request."""

    question: str = Field(..., min_length=1, max_length=2000)
    project_id: UUID | None = None
    document_ids: list[UUID] | None = None
    top_k: int = Field(default=5, ge=1, le=20)


class SourceResponse(BaseModel):
    """Source reference in query response."""

    document_id: UUID
    filename: str
    page_number: int | None
    excerpt: str
    score: float


class QueryResponse(BaseModel):
    """RAG query response."""

    answer: str
    sources: list[SourceResponse]
    query: str


# ============================================================================
# Health Schemas
# ============================================================================


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database: str
