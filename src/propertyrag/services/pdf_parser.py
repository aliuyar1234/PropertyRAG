"""PDF parsing service using pdfplumber."""

from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from propertyrag.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ParsedPage:
    """A parsed page from a PDF."""

    page_number: int
    text: str


@dataclass
class ParsedDocument:
    """A fully parsed PDF document."""

    filename: str
    pages: list[ParsedPage]
    page_count: int

    @property
    def full_text(self) -> str:
        """Get the full text of the document."""
        return "\n\n".join(page.text for page in self.pages)


class PDFParserError(Exception):
    """Error during PDF parsing."""

    pass


class PDFParser:
    """Service for parsing PDF documents."""

    def parse(self, file_path: Path) -> ParsedDocument:
        """
        Parse a PDF file and extract text from all pages.

        Args:
            file_path: Path to the PDF file.

        Returns:
            ParsedDocument with extracted text per page.

        Raises:
            PDFParserError: If parsing fails.
        """
        logger.info("parsing_pdf", file_path=str(file_path))

        if not file_path.exists():
            raise PDFParserError(f"File not found: {file_path}")

        if not file_path.suffix.lower() == ".pdf":
            raise PDFParserError(f"Not a PDF file: {file_path}")

        try:
            pages: list[ParsedPage] = []

            with pdfplumber.open(file_path) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = self._extract_page_text(page)
                    pages.append(ParsedPage(page_number=i, text=text))

                page_count = len(pdf.pages)

            logger.info(
                "pdf_parsed",
                file_path=str(file_path),
                page_count=page_count,
                total_chars=sum(len(p.text) for p in pages),
            )

            return ParsedDocument(
                filename=file_path.name,
                pages=pages,
                page_count=page_count,
            )

        except pdfplumber.pdfminer.pdfparser.PDFSyntaxError as e:
            logger.error("pdf_syntax_error", file_path=str(file_path), error=str(e))
            raise PDFParserError(f"Invalid PDF syntax: {e}") from e
        except Exception as e:
            logger.error("pdf_parse_error", file_path=str(file_path), error=str(e))
            raise PDFParserError(f"Failed to parse PDF: {e}") from e

    def _extract_page_text(self, page: pdfplumber.page.Page) -> str:
        """
        Extract text from a single page.

        Handles tables specially for better extraction.
        """
        # Extract tables first
        tables = page.extract_tables()
        table_texts = []

        for table in tables:
            if table:
                # Convert table to text representation
                rows = []
                for row in table:
                    # Filter None values and join cells
                    cells = [str(cell) if cell else "" for cell in row]
                    rows.append(" | ".join(cells))
                table_texts.append("\n".join(rows))

        # Extract regular text
        text = page.extract_text() or ""

        # If we found tables, append them (they might be duplicated in text,
        # but that's better than missing them)
        if table_texts:
            text = text + "\n\n" + "\n\n".join(table_texts)

        # Clean up the text
        text = self._clean_text(text)

        return text

    def _clean_text(self, text: str) -> str:
        """Clean extracted text."""
        # Replace multiple whitespaces with single space
        import re

        text = re.sub(r"[ \t]+", " ", text)

        # Replace multiple newlines with double newline
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Strip leading/trailing whitespace
        text = text.strip()

        return text

    def parse_bytes(self, content: bytes, filename: str) -> ParsedDocument:
        """
        Parse PDF from bytes content.

        Args:
            content: PDF file content as bytes.
            filename: Original filename.

        Returns:
            ParsedDocument with extracted text per page.
        """
        import io

        logger.info("parsing_pdf_bytes", filename=filename, size=len(content))

        try:
            pages: list[ParsedPage] = []

            with pdfplumber.open(io.BytesIO(content)) as pdf:
                for i, page in enumerate(pdf.pages, start=1):
                    text = self._extract_page_text(page)
                    pages.append(ParsedPage(page_number=i, text=text))

                page_count = len(pdf.pages)

            logger.info(
                "pdf_parsed",
                filename=filename,
                page_count=page_count,
                total_chars=sum(len(p.text) for p in pages),
            )

            return ParsedDocument(
                filename=filename,
                pages=pages,
                page_count=page_count,
            )

        except Exception as e:
            logger.error("pdf_parse_error", filename=filename, error=str(e))
            raise PDFParserError(f"Failed to parse PDF: {e}") from e
