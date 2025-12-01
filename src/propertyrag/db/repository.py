"""Repository pattern for database operations."""

from uuid import UUID

from sqlalchemy import func, select, text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from propertyrag.core.models import DocumentType, ProcessingStatus
from propertyrag.db.models import (
    ChunkModel,
    DocumentModel,
    ExtractedDataModel,
    ProjectModel,
)


class ProjectRepository:
    """Repository for project operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(self, name: str, description: str | None = None) -> ProjectModel:
        """Create a new project."""
        project = ProjectModel(name=name, description=description)
        self.session.add(project)
        await self.session.flush()
        return project

    async def get_by_id(self, project_id: UUID) -> ProjectModel | None:
        """Get a project by ID."""
        return await self.session.get(ProjectModel, project_id)

    async def get_all(self) -> list[ProjectModel]:
        """Get all projects."""
        result = await self.session.execute(
            select(ProjectModel).order_by(ProjectModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def delete(self, project_id: UUID) -> bool:
        """Delete a project by ID."""
        project = await self.get_by_id(project_id)
        if project:
            await self.session.delete(project)
            return True
        return False


class DocumentRepository:
    """Repository for document operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        filename: str,
        document_type: DocumentType = DocumentType.UNKNOWN,
        project_id: UUID | None = None,
    ) -> DocumentModel:
        """Create a new document."""
        document = DocumentModel(
            filename=filename,
            document_type=document_type,
            project_id=project_id,
        )
        self.session.add(document)
        await self.session.flush()
        return document

    async def get_by_id(
        self, document_id: UUID, include_chunks: bool = False
    ) -> DocumentModel | None:
        """Get a document by ID."""
        if include_chunks:
            result = await self.session.execute(
                select(DocumentModel)
                .options(selectinload(DocumentModel.chunks))
                .where(DocumentModel.id == document_id)
            )
            return result.scalar_one_or_none()
        return await self.session.get(DocumentModel, document_id)

    async def get_by_project(self, project_id: UUID) -> list[DocumentModel]:
        """Get all documents in a project."""
        result = await self.session.execute(
            select(DocumentModel)
            .where(DocumentModel.project_id == project_id)
            .order_by(DocumentModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def get_all(self) -> list[DocumentModel]:
        """Get all documents."""
        result = await self.session.execute(
            select(DocumentModel).order_by(DocumentModel.created_at.desc())
        )
        return list(result.scalars().all())

    async def update_status(
        self,
        document_id: UUID,
        status: ProcessingStatus,
        page_count: int | None = None,
    ) -> DocumentModel | None:
        """Update document processing status."""
        document = await self.get_by_id(document_id)
        if document:
            document.status = status
            if page_count is not None:
                document.page_count = page_count
            await self.session.flush()
        return document

    async def update_type(
        self, document_id: UUID, document_type: DocumentType
    ) -> DocumentModel | None:
        """Update document type."""
        document = await self.get_by_id(document_id)
        if document:
            document.document_type = document_type
            await self.session.flush()
        return document

    async def delete(self, document_id: UUID) -> bool:
        """Delete a document by ID."""
        document = await self.get_by_id(document_id)
        if document:
            await self.session.delete(document)
            return True
        return False


class ChunkRepository:
    """Repository for chunk operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create_many(
        self,
        document_id: UUID,
        chunks: list[dict],
    ) -> list[ChunkModel]:
        """Create multiple chunks for a document."""
        chunk_models = [
            ChunkModel(
                document_id=document_id,
                content=chunk["content"],
                page_number=chunk.get("page_number"),
                chunk_index=chunk["chunk_index"],
                token_count=chunk["token_count"],
                embedding=chunk["embedding"],
            )
            for chunk in chunks
        ]
        self.session.add_all(chunk_models)
        await self.session.flush()
        return chunk_models

    async def get_by_document(self, document_id: UUID) -> list[ChunkModel]:
        """Get all chunks for a document."""
        result = await self.session.execute(
            select(ChunkModel)
            .where(ChunkModel.document_id == document_id)
            .order_by(ChunkModel.chunk_index)
        )
        return list(result.scalars().all())

    async def search_similar(
        self,
        embedding: list[float],
        top_k: int = 5,
        project_id: UUID | None = None,
        document_ids: list[UUID] | None = None,
    ) -> list[tuple[ChunkModel, float]]:
        """Search for similar chunks using cosine similarity."""
        # Build the base query with similarity score
        embedding_str = f"[{','.join(str(x) for x in embedding)}]"

        query = (
            select(
                ChunkModel,
                (1 - ChunkModel.embedding.cosine_distance(text(f"'{embedding_str}'::vector"))).label(
                    "similarity"
                ),
            )
            .join(DocumentModel)
        )

        # Apply filters
        if document_ids:
            query = query.where(ChunkModel.document_id.in_(document_ids))
        elif project_id:
            query = query.where(DocumentModel.project_id == project_id)

        # Order by similarity and limit
        query = query.order_by(text("similarity DESC")).limit(top_k)

        result = await self.session.execute(query)
        return [(row[0], row[1]) for row in result.all()]


class ExtractedDataRepository:
    """Repository for extracted data operations."""

    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def create(
        self,
        document_id: UUID,
        document_type: DocumentType,
        data: dict,
        confidence: float | None = None,
    ) -> ExtractedDataModel:
        """Create extracted data for a document."""
        extracted = ExtractedDataModel(
            document_id=document_id,
            document_type=document_type,
            data=data,
            extraction_confidence=confidence,
        )
        self.session.add(extracted)
        await self.session.flush()
        return extracted

    async def get_by_document(self, document_id: UUID) -> ExtractedDataModel | None:
        """Get extracted data for a document."""
        result = await self.session.execute(
            select(ExtractedDataModel).where(
                ExtractedDataModel.document_id == document_id
            )
        )
        return result.scalar_one_or_none()

    async def update(
        self,
        document_id: UUID,
        data: dict,
        confidence: float | None = None,
    ) -> ExtractedDataModel | None:
        """Update extracted data for a document."""
        extracted = await self.get_by_document(document_id)
        if extracted:
            extracted.data = data
            if confidence is not None:
                extracted.extraction_confidence = confidence
            await self.session.flush()
        return extracted
