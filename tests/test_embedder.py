"""Tests for the embedder service."""

import pytest
from unittest.mock import AsyncMock, MagicMock

from propertyrag.services.embedder import Embedder, EmbeddingError
from propertyrag.services.chunker import TextChunk


class TestEmbedder:
    """Tests for the Embedder class."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock OpenAI client."""
        client = AsyncMock()

        # Mock single embedding response
        response = MagicMock()
        response.data = [MagicMock(embedding=[0.1] * 1536, index=0)]
        client.embeddings.create = AsyncMock(return_value=response)

        return client

    @pytest.fixture
    def embedder(self, mock_client: AsyncMock) -> Embedder:
        """Create an embedder with mock client."""
        return Embedder(client=mock_client)

    @pytest.mark.asyncio
    async def test_embed_text(self, embedder: Embedder) -> None:
        """Test embedding a single text."""
        embedding = await embedder.embed_text("Hello world")

        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

    @pytest.mark.asyncio
    async def test_embed_texts(self, embedder: Embedder, mock_client: AsyncMock) -> None:
        """Test embedding multiple texts."""
        # Setup mock for batch response
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
            MagicMock(embedding=[0.2] * 1536, index=1),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=response)

        embeddings = await embedder.embed_texts(["Text 1", "Text 2"])

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 1536
        assert len(embeddings[1]) == 1536

    @pytest.mark.asyncio
    async def test_embed_empty_list(self, embedder: Embedder) -> None:
        """Test embedding an empty list."""
        embeddings = await embedder.embed_texts([])

        assert embeddings == []

    @pytest.mark.asyncio
    async def test_embed_chunks(self, embedder: Embedder, mock_client: AsyncMock) -> None:
        """Test embedding chunks."""
        chunks = [
            TextChunk(content="Chunk 1", page_number=1, chunk_index=0, token_count=10),
            TextChunk(content="Chunk 2", page_number=1, chunk_index=1, token_count=10),
        ]

        # Setup mock for batch response
        response = MagicMock()
        response.data = [
            MagicMock(embedding=[0.1] * 1536, index=0),
            MagicMock(embedding=[0.2] * 1536, index=1),
        ]
        mock_client.embeddings.create = AsyncMock(return_value=response)

        result = await embedder.embed_chunks(chunks)

        assert len(result) == 2
        assert result[0][0] == chunks[0]
        assert len(result[0][1]) == 1536

    @pytest.mark.asyncio
    async def test_embed_error_handling(self, embedder: Embedder, mock_client: AsyncMock) -> None:
        """Test error handling during embedding."""
        mock_client.embeddings.create = AsyncMock(side_effect=Exception("API Error"))

        with pytest.raises(EmbeddingError):
            await embedder.embed_text("Test")
