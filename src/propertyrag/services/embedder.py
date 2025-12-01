"""Embedding service using OpenAI API."""

from openai import AsyncOpenAI

from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger
from propertyrag.services.chunker import TextChunk

logger = get_logger(__name__)

# OpenAI has a limit of 8191 tokens per embedding request for text-embedding-3-small
MAX_TOKENS_PER_REQUEST = 8191
# Maximum batch size for embeddings API
MAX_BATCH_SIZE = 2048


class EmbeddingError(Exception):
    """Error during embedding generation."""

    pass


class Embedder:
    """Service for generating embeddings using OpenAI."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        """
        Initialize the embedder.

        Args:
            client: Optional AsyncOpenAI client. If not provided, creates one.
        """
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_embedding_model
        self.dimensions = settings.embedding_dimensions

        logger.info(
            "embedder_initialized",
            model=self.model,
            dimensions=self.dimensions,
        )

    async def embed_text(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed.

        Returns:
            Embedding vector as list of floats.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        try:
            response = await self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimensions,
            )
            return response.data[0].embedding

        except Exception as e:
            logger.error("embedding_error", error=str(e), text_length=len(text))
            raise EmbeddingError(f"Failed to generate embedding: {e}") from e

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for multiple texts.

        Handles batching automatically to stay within API limits.

        Args:
            texts: List of texts to embed.

        Returns:
            List of embedding vectors.

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not texts:
            return []

        logger.info("embedding_texts", count=len(texts))

        try:
            # Process in batches
            all_embeddings: list[list[float]] = []

            for i in range(0, len(texts), MAX_BATCH_SIZE):
                batch = texts[i : i + MAX_BATCH_SIZE]

                response = await self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )

                # Sort by index to maintain order
                sorted_data = sorted(response.data, key=lambda x: x.index)
                batch_embeddings = [item.embedding for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

                logger.debug(
                    "batch_embedded",
                    batch_start=i,
                    batch_size=len(batch),
                )

            logger.info(
                "texts_embedded",
                count=len(texts),
                embedding_count=len(all_embeddings),
            )

            return all_embeddings

        except Exception as e:
            logger.error("embedding_batch_error", error=str(e), text_count=len(texts))
            raise EmbeddingError(f"Failed to generate embeddings: {e}") from e

    async def embed_chunks(
        self, chunks: list[TextChunk]
    ) -> list[tuple[TextChunk, list[float]]]:
        """
        Generate embeddings for a list of chunks.

        Args:
            chunks: List of text chunks.

        Returns:
            List of tuples (chunk, embedding).

        Raises:
            EmbeddingError: If embedding generation fails.
        """
        if not chunks:
            return []

        logger.info("embedding_chunks", count=len(chunks))

        texts = [chunk.content for chunk in chunks]
        embeddings = await self.embed_texts(texts)

        return list(zip(chunks, embeddings, strict=True))
