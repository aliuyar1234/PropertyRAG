"""Tests for the chunker service."""

import pytest

from propertyrag.services.chunker import Chunker, TextChunk
from propertyrag.services.pdf_parser import ParsedDocument, ParsedPage


class TestChunker:
    """Tests for the Chunker class."""

    @pytest.fixture
    def chunker(self) -> Chunker:
        """Create a chunker with small chunk size for testing."""
        return Chunker(chunk_size=50, chunk_overlap=10)

    def test_count_tokens(self, chunker: Chunker) -> None:
        """Test token counting."""
        text = "Hello world, this is a test."
        token_count = chunker.count_tokens(text)

        assert token_count > 0
        assert isinstance(token_count, int)

    def test_chunk_short_document(self, chunker: Chunker) -> None:
        """Test chunking a short document that fits in one chunk."""
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[ParsedPage(page_number=1, text="Short text.")],
            page_count=1,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 1
        assert chunks[0].content == "Short text."
        assert chunks[0].page_number == 1
        assert chunks[0].chunk_index == 0

    def test_chunk_long_document(self, chunker: Chunker) -> None:
        """Test chunking a document that needs multiple chunks."""
        long_text = " ".join(["This is a test sentence."] * 50)
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[ParsedPage(page_number=1, text=long_text)],
            page_count=1,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) > 1
        # Check indices are sequential
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_multiple_pages(self, chunker: Chunker) -> None:
        """Test chunking a document with multiple pages."""
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[
                ParsedPage(page_number=1, text="Page one content."),
                ParsedPage(page_number=2, text="Page two content."),
            ],
            page_count=2,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) >= 2
        # Check page numbers are preserved
        page_numbers = {chunk.page_number for chunk in chunks}
        assert 1 in page_numbers
        assert 2 in page_numbers

    def test_chunk_empty_document(self, chunker: Chunker) -> None:
        """Test chunking an empty document."""
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[ParsedPage(page_number=1, text="")],
            page_count=1,
        )

        chunks = chunker.chunk_document(doc)

        assert len(chunks) == 0

    def test_chunk_preserves_paragraphs(self, chunker: Chunker) -> None:
        """Test that chunking respects paragraph boundaries when possible."""
        text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[ParsedPage(page_number=1, text=text)],
            page_count=1,
        )

        chunks = chunker.chunk_document(doc)

        # Should have chunked respecting paragraph boundaries
        assert len(chunks) >= 1
        # Content should be present
        full_content = " ".join(c.content for c in chunks)
        assert "First paragraph" in full_content
        assert "Second paragraph" in full_content

    def test_token_count_accuracy(self, chunker: Chunker) -> None:
        """Test that token counts are accurate."""
        doc = ParsedDocument(
            filename="test.pdf",
            pages=[ParsedPage(page_number=1, text="Hello world test.")],
            page_count=1,
        )

        chunks = chunker.chunk_document(doc)

        for chunk in chunks:
            actual_tokens = chunker.count_tokens(chunk.content)
            assert chunk.token_count == actual_tokens
