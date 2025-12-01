"""Document ingestion pipeline."""

from pathlib import Path
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession

from propertyrag.core.logging import get_logger
from propertyrag.core.models import DocumentType, ProcessingStatus
from propertyrag.db.repository import (
    ChunkRepository,
    DocumentRepository,
    ExtractedDataRepository,
)
from propertyrag.services.chunker import Chunker
from propertyrag.services.classifier import DocumentClassifier
from propertyrag.services.embedder import Embedder
from propertyrag.services.extractor import DataExtractor, ExtractionError
from propertyrag.services.pdf_parser import PDFParser, PDFParserError

logger = get_logger(__name__)


class IngestionError(Exception):
    """Error during document ingestion."""

    pass


class IngestionPipeline:
    """
    Pipeline for ingesting PDF documents into the vector store.

    Flow:
    1. Create document record (PENDING)
    2. Parse PDF to extract text
    3. Classify document type (if unknown)
    4. Chunk text into smaller pieces
    5. Generate embeddings for chunks
    6. Store chunks with embeddings
    7. Extract structured data
    8. Update document status (COMPLETED/FAILED)
    """

    def __init__(
        self,
        session: AsyncSession,
        pdf_parser: PDFParser | None = None,
        chunker: Chunker | None = None,
        embedder: Embedder | None = None,
        classifier: DocumentClassifier | None = None,
        extractor: DataExtractor | None = None,
    ) -> None:
        """
        Initialize the ingestion pipeline.

        Args:
            session: Database session.
            pdf_parser: Optional PDF parser. Creates default if not provided.
            chunker: Optional chunker. Creates default if not provided.
            embedder: Optional embedder. Creates default if not provided.
            classifier: Optional classifier. Creates default if not provided.
            extractor: Optional extractor. Creates default if not provided.
        """
        self.session = session
        self.pdf_parser = pdf_parser or PDFParser()
        self.chunker = chunker or Chunker()
        self.embedder = embedder or Embedder()
        self.classifier = classifier or DocumentClassifier()
        self.extractor = extractor or DataExtractor()

        self.doc_repo = DocumentRepository(session)
        self.chunk_repo = ChunkRepository(session)
        self.extracted_repo = ExtractedDataRepository(session)

    async def ingest_file(
        self,
        file_path: Path,
        document_type: DocumentType = DocumentType.UNKNOWN,
        project_id: UUID | None = None,
        auto_extract: bool = True,
    ) -> UUID:
        """
        Ingest a PDF file into the system.

        Args:
            file_path: Path to the PDF file.
            document_type: Type of document. Defaults to UNKNOWN (will be classified).
            project_id: Optional project ID to associate with.
            auto_extract: Whether to automatically extract structured data.

        Returns:
            The document ID.

        Raises:
            IngestionError: If ingestion fails.
        """
        logger.info(
            "ingestion_started",
            file_path=str(file_path),
            document_type=document_type.value,
        )

        # Create document record
        document = await self.doc_repo.create(
            filename=file_path.name,
            document_type=document_type,
            project_id=project_id,
        )
        document_id = document.id

        try:
            # Update status to processing
            await self.doc_repo.update_status(document_id, ProcessingStatus.PROCESSING)

            # Parse PDF
            parsed_doc = self.pdf_parser.parse(file_path)
            full_text = parsed_doc.full_text

            # Classify document if type is unknown
            if document_type == DocumentType.UNKNOWN:
                document_type = await self.classifier.classify(full_text)
                await self.doc_repo.update_type(document_id, document_type)
                logger.info(
                    "document_classified",
                    document_id=str(document_id),
                    document_type=document_type.value,
                )

            # Chunk the document
            chunks = self.chunker.chunk_document(parsed_doc)

            if not chunks:
                logger.warning("no_chunks_created", document_id=str(document_id))
                await self.doc_repo.update_status(
                    document_id,
                    ProcessingStatus.COMPLETED,
                    page_count=parsed_doc.page_count,
                )
                return document_id

            # Generate embeddings
            chunks_with_embeddings = await self.embedder.embed_chunks(chunks)

            # Store chunks
            chunk_data = [
                {
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "embedding": embedding,
                }
                for chunk, embedding in chunks_with_embeddings
            ]

            await self.chunk_repo.create_many(document_id, chunk_data)

            # Extract structured data if enabled and document type is known
            if auto_extract and document_type != DocumentType.UNKNOWN:
                await self._extract_and_store(document_id, full_text, document_type)

            # Update status to completed
            await self.doc_repo.update_status(
                document_id,
                ProcessingStatus.COMPLETED,
                page_count=parsed_doc.page_count,
            )

            logger.info(
                "ingestion_completed",
                document_id=str(document_id),
                document_type=document_type.value,
                chunk_count=len(chunks),
                page_count=parsed_doc.page_count,
            )

            return document_id

        except PDFParserError as e:
            logger.error(
                "ingestion_parse_error",
                document_id=str(document_id),
                error=str(e),
            )
            await self.doc_repo.update_status(document_id, ProcessingStatus.FAILED)
            raise IngestionError(f"Failed to parse PDF: {e}") from e

        except Exception as e:
            logger.error(
                "ingestion_error",
                document_id=str(document_id),
                error=str(e),
            )
            await self.doc_repo.update_status(document_id, ProcessingStatus.FAILED)
            raise IngestionError(f"Ingestion failed: {e}") from e

    async def ingest_bytes(
        self,
        content: bytes,
        filename: str,
        document_type: DocumentType = DocumentType.UNKNOWN,
        project_id: UUID | None = None,
        auto_extract: bool = True,
    ) -> UUID:
        """
        Ingest a PDF from bytes content.

        Args:
            content: PDF file content as bytes.
            filename: Original filename.
            document_type: Type of document. Defaults to UNKNOWN (will be classified).
            project_id: Optional project ID to associate with.
            auto_extract: Whether to automatically extract structured data.

        Returns:
            The document ID.

        Raises:
            IngestionError: If ingestion fails.
        """
        logger.info(
            "ingestion_bytes_started",
            filename=filename,
            size=len(content),
            document_type=document_type.value,
        )

        # Create document record
        document = await self.doc_repo.create(
            filename=filename,
            document_type=document_type,
            project_id=project_id,
        )
        document_id = document.id

        try:
            # Update status to processing
            await self.doc_repo.update_status(document_id, ProcessingStatus.PROCESSING)

            # Parse PDF from bytes
            parsed_doc = self.pdf_parser.parse_bytes(content, filename)
            full_text = parsed_doc.full_text

            # Classify document if type is unknown
            if document_type == DocumentType.UNKNOWN:
                document_type = await self.classifier.classify(full_text)
                await self.doc_repo.update_type(document_id, document_type)
                logger.info(
                    "document_classified",
                    document_id=str(document_id),
                    document_type=document_type.value,
                )

            # Chunk the document
            chunks = self.chunker.chunk_document(parsed_doc)

            if not chunks:
                logger.warning("no_chunks_created", document_id=str(document_id))
                await self.doc_repo.update_status(
                    document_id,
                    ProcessingStatus.COMPLETED,
                    page_count=parsed_doc.page_count,
                )
                return document_id

            # Generate embeddings
            chunks_with_embeddings = await self.embedder.embed_chunks(chunks)

            # Store chunks
            chunk_data = [
                {
                    "content": chunk.content,
                    "page_number": chunk.page_number,
                    "chunk_index": chunk.chunk_index,
                    "token_count": chunk.token_count,
                    "embedding": embedding,
                }
                for chunk, embedding in chunks_with_embeddings
            ]

            await self.chunk_repo.create_many(document_id, chunk_data)

            # Extract structured data if enabled and document type is known
            if auto_extract and document_type != DocumentType.UNKNOWN:
                await self._extract_and_store(document_id, full_text, document_type)

            # Update status to completed
            await self.doc_repo.update_status(
                document_id,
                ProcessingStatus.COMPLETED,
                page_count=parsed_doc.page_count,
            )

            logger.info(
                "ingestion_completed",
                document_id=str(document_id),
                document_type=document_type.value,
                chunk_count=len(chunks),
                page_count=parsed_doc.page_count,
            )

            return document_id

        except PDFParserError as e:
            logger.error(
                "ingestion_parse_error",
                document_id=str(document_id),
                error=str(e),
            )
            await self.doc_repo.update_status(document_id, ProcessingStatus.FAILED)
            raise IngestionError(f"Failed to parse PDF: {e}") from e

        except Exception as e:
            logger.error(
                "ingestion_error",
                document_id=str(document_id),
                error=str(e),
            )
            await self.doc_repo.update_status(document_id, ProcessingStatus.FAILED)
            raise IngestionError(f"Ingestion failed: {e}") from e

    async def _extract_and_store(
        self,
        document_id: UUID,
        text: str,
        document_type: DocumentType,
    ) -> None:
        """
        Extract structured data and store it.

        This is a best-effort operation - extraction failures don't fail the ingestion.
        """
        try:
            extracted_data, confidence = await self.extractor.extract(
                text, document_type
            )

            await self.extracted_repo.create(
                document_id=document_id,
                document_type=document_type,
                data=extracted_data.model_dump(mode="json"),
                confidence=confidence,
            )

            logger.info(
                "extraction_stored",
                document_id=str(document_id),
                document_type=document_type.value,
                confidence=confidence,
            )

        except ExtractionError as e:
            # Log but don't fail - extraction is optional
            logger.warning(
                "extraction_failed",
                document_id=str(document_id),
                error=str(e),
            )

    async def extract_document(
        self,
        document_id: UUID,
        force: bool = False,
    ) -> dict | None:
        """
        Extract or re-extract structured data for an existing document.

        Args:
            document_id: ID of the document to extract.
            force: If True, re-extract even if data already exists.

        Returns:
            Extracted data as dictionary, or None if extraction failed.
        """
        # Check if extraction already exists
        existing = await self.extracted_repo.get_by_document(document_id)
        if existing and not force:
            logger.info(
                "extraction_exists",
                document_id=str(document_id),
            )
            return existing.data

        # Get document and its text
        document = await self.doc_repo.get_by_id(document_id, include_chunks=True)
        if not document:
            logger.error("document_not_found", document_id=str(document_id))
            return None

        if document.document_type == DocumentType.UNKNOWN:
            logger.warning(
                "cannot_extract_unknown_type",
                document_id=str(document_id),
            )
            return None

        # Reconstruct text from chunks
        chunks = sorted(document.chunks, key=lambda c: c.chunk_index)
        text = "\n\n".join(chunk.content for chunk in chunks)

        try:
            extracted_data, confidence = await self.extractor.extract(
                text, document.document_type
            )
            data_dict = extracted_data.model_dump(mode="json")

            if existing:
                await self.extracted_repo.update(
                    document_id=document_id,
                    data=data_dict,
                    confidence=confidence,
                )
            else:
                await self.extracted_repo.create(
                    document_id=document_id,
                    document_type=document.document_type,
                    data=data_dict,
                    confidence=confidence,
                )

            logger.info(
                "document_extracted",
                document_id=str(document_id),
                confidence=confidence,
            )

            return data_dict

        except ExtractionError as e:
            logger.error(
                "document_extraction_failed",
                document_id=str(document_id),
                error=str(e),
            )
            return None
