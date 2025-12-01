"""Text chunking service with token-based splitting."""

from dataclasses import dataclass

import tiktoken

from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger
from propertyrag.services.pdf_parser import ParsedDocument, ParsedPage

logger = get_logger(__name__)


@dataclass
class TextChunk:
    """A chunk of text with metadata."""

    content: str
    page_number: int | None
    chunk_index: int
    token_count: int


class Chunker:
    """
    Service for splitting documents into chunks.

    Uses tiktoken for accurate token counting compatible with OpenAI models.
    Implements semantic chunking that respects paragraph boundaries when possible.
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        """
        Initialize the chunker.

        Args:
            chunk_size: Maximum tokens per chunk. Defaults to settings value.
            chunk_overlap: Overlap tokens between chunks. Defaults to settings value.
        """
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        # Use cl100k_base encoding (used by text-embedding-3-small and GPT-4)
        self.encoding = tiktoken.get_encoding("cl100k_base")

        logger.info(
            "chunker_initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def count_tokens(self, text: str) -> int:
        """Count tokens in a text string."""
        return len(self.encoding.encode(text))

    def chunk_document(self, document: ParsedDocument) -> list[TextChunk]:
        """
        Split a parsed document into chunks.

        Uses a page-aware strategy that:
        1. Processes each page separately to maintain page references
        2. Splits on paragraph boundaries when possible
        3. Falls back to sentence boundaries for large paragraphs
        4. Uses token-based splitting as last resort

        Args:
            document: Parsed PDF document.

        Returns:
            List of text chunks with metadata.
        """
        logger.info("chunking_document", filename=document.filename)

        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in document.pages:
            page_chunks = self._chunk_page(page, chunk_index)
            chunks.extend(page_chunks)
            chunk_index += len(page_chunks)

        logger.info(
            "document_chunked",
            filename=document.filename,
            chunk_count=len(chunks),
            total_tokens=sum(c.token_count for c in chunks),
        )

        return chunks

    def _chunk_page(self, page: ParsedPage, start_index: int) -> list[TextChunk]:
        """Split a single page into chunks."""
        if not page.text.strip():
            return []

        # Split into paragraphs first
        paragraphs = self._split_into_paragraphs(page.text)

        chunks: list[TextChunk] = []
        current_text = ""
        current_tokens = 0
        chunk_index = start_index

        for paragraph in paragraphs:
            para_tokens = self.count_tokens(paragraph)

            # If paragraph alone exceeds chunk size, split it further
            if para_tokens > self.chunk_size:
                # First, save any accumulated text
                if current_text.strip():
                    chunks.append(
                        TextChunk(
                            content=current_text.strip(),
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1
                    current_text = ""
                    current_tokens = 0

                # Split the large paragraph
                para_chunks = self._split_large_text(
                    paragraph, page.page_number, chunk_index
                )
                chunks.extend(para_chunks)
                chunk_index += len(para_chunks)

            # If adding this paragraph exceeds limit, start new chunk
            elif current_tokens + para_tokens > self.chunk_size:
                if current_text.strip():
                    chunks.append(
                        TextChunk(
                            content=current_text.strip(),
                            page_number=page.page_number,
                            chunk_index=chunk_index,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1

                    # Add overlap from end of previous chunk
                    overlap_text = self._get_overlap_text(current_text)
                    current_text = overlap_text + "\n\n" + paragraph
                    current_tokens = self.count_tokens(current_text)
                else:
                    current_text = paragraph
                    current_tokens = para_tokens

            # Otherwise, accumulate
            else:
                if current_text:
                    current_text += "\n\n" + paragraph
                else:
                    current_text = paragraph
                current_tokens = self.count_tokens(current_text)

        # Don't forget the last chunk
        if current_text.strip():
            chunks.append(
                TextChunk(
                    content=current_text.strip(),
                    page_number=page.page_number,
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(current_text.strip()),
                )
            )

        return chunks

    def _split_into_paragraphs(self, text: str) -> list[str]:
        """Split text into paragraphs."""
        # Split on double newlines
        paragraphs = text.split("\n\n")

        # Clean up and filter empty paragraphs
        return [p.strip() for p in paragraphs if p.strip()]

    def _split_large_text(
        self, text: str, page_number: int | None, start_index: int
    ) -> list[TextChunk]:
        """
        Split text that exceeds chunk size.

        First tries sentence boundaries, then falls back to token-based splitting.
        """
        chunks: list[TextChunk] = []

        # Try to split on sentences first
        sentences = self._split_into_sentences(text)

        current_text = ""
        current_tokens = 0
        chunk_index = start_index

        for sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # If single sentence exceeds chunk size, split by tokens
            if sentence_tokens > self.chunk_size:
                if current_text.strip():
                    chunks.append(
                        TextChunk(
                            content=current_text.strip(),
                            page_number=page_number,
                            chunk_index=chunk_index,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1
                    current_text = ""
                    current_tokens = 0

                # Force split by tokens
                token_chunks = self._force_split_by_tokens(
                    sentence, page_number, chunk_index
                )
                chunks.extend(token_chunks)
                chunk_index += len(token_chunks)

            elif current_tokens + sentence_tokens > self.chunk_size:
                if current_text.strip():
                    chunks.append(
                        TextChunk(
                            content=current_text.strip(),
                            page_number=page_number,
                            chunk_index=chunk_index,
                            token_count=current_tokens,
                        )
                    )
                    chunk_index += 1

                    overlap_text = self._get_overlap_text(current_text)
                    current_text = overlap_text + " " + sentence
                    current_tokens = self.count_tokens(current_text)
                else:
                    current_text = sentence
                    current_tokens = sentence_tokens
            else:
                if current_text:
                    current_text += " " + sentence
                else:
                    current_text = sentence
                current_tokens = self.count_tokens(current_text)

        if current_text.strip():
            chunks.append(
                TextChunk(
                    content=current_text.strip(),
                    page_number=page_number,
                    chunk_index=chunk_index,
                    token_count=self.count_tokens(current_text.strip()),
                )
            )

        return chunks

    def _split_into_sentences(self, text: str) -> list[str]:
        """Split text into sentences."""
        import re

        # Simple sentence splitting - handles common cases
        # Splits on . ! ? followed by space and capital letter or end of string
        sentence_pattern = r"(?<=[.!?])\s+(?=[A-ZÄÖÜ])|(?<=[.!?])$"
        sentences = re.split(sentence_pattern, text)

        return [s.strip() for s in sentences if s.strip()]

    def _force_split_by_tokens(
        self, text: str, page_number: int | None, start_index: int
    ) -> list[TextChunk]:
        """Force split text by token count when no natural boundaries work."""
        chunks: list[TextChunk] = []
        tokens = self.encoding.encode(text)

        chunk_index = start_index
        i = 0

        while i < len(tokens):
            # Take chunk_size tokens
            end = min(i + self.chunk_size, len(tokens))
            chunk_tokens = tokens[i:end]
            chunk_text = self.encoding.decode(chunk_tokens)

            chunks.append(
                TextChunk(
                    content=chunk_text,
                    page_number=page_number,
                    chunk_index=chunk_index,
                    token_count=len(chunk_tokens),
                )
            )

            # Move forward, accounting for overlap
            i = end - self.chunk_overlap if end < len(tokens) else end
            chunk_index += 1

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """Get the last N tokens of text for overlap."""
        tokens = self.encoding.encode(text)
        if len(tokens) <= self.chunk_overlap:
            return text

        overlap_tokens = tokens[-self.chunk_overlap :]
        return self.encoding.decode(overlap_tokens)
