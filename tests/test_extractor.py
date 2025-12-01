"""Tests for the extractor service."""

import json
import pytest
from unittest.mock import AsyncMock, MagicMock

from propertyrag.core.models import DocumentType, MietvertragData
from propertyrag.services.extractor import DataExtractor, ExtractionError


class TestDataExtractor:
    """Tests for the DataExtractor class."""

    @pytest.fixture
    def mock_client(self) -> AsyncMock:
        """Create a mock OpenAI client."""
        client = AsyncMock()
        return client

    @pytest.fixture
    def extractor(self, mock_client: AsyncMock) -> DataExtractor:
        """Create an extractor with mock client."""
        return DataExtractor(client=mock_client)

    @pytest.mark.asyncio
    async def test_extract_mietvertrag(
        self, extractor: DataExtractor, mock_client: AsyncMock
    ) -> None:
        """Test extracting data from a rental contract."""
        # Setup mock response
        mock_data = {
            "vermieter": {"name": "Max Mustermann", "adresse": "Musterstr. 1"},
            "mieter": {"name": "Erika Musterfrau", "adresse": "Beispielweg 2"},
            "objekt_adresse": "Teststraße 123, 12345 Berlin",
            "nettomiete_eur": 1000,
            "nebenkosten_eur": 200,
            "mietbeginn": "2024-01-01",
        }

        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=json.dumps(mock_data)))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        data, confidence = await extractor.extract(
            "Sample rental contract text", DocumentType.MIETVERTRAG
        )

        assert isinstance(data, MietvertragData)
        assert data.nettomiete_eur == 1000
        assert data.nebenkosten_eur == 200
        assert confidence > 0

    @pytest.mark.asyncio
    async def test_extract_unknown_type(self, extractor: DataExtractor) -> None:
        """Test that extracting from unknown type raises error."""
        with pytest.raises(ExtractionError):
            await extractor.extract("Some text", DocumentType.UNKNOWN)

    @pytest.mark.asyncio
    async def test_extract_handles_partial_data(
        self, extractor: DataExtractor, mock_client: AsyncMock
    ) -> None:
        """Test extraction with partial/incomplete data."""
        # Only some fields filled
        mock_data = {
            "nettomiete_eur": 1500,
            "objekt_adresse": "Teststr. 1",
        }

        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=json.dumps(mock_data)))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        data, confidence = await extractor.extract(
            "Partial contract text", DocumentType.MIETVERTRAG
        )

        assert isinstance(data, MietvertragData)
        assert data.nettomiete_eur == 1500
        assert data.vermieter is None  # Not provided
        # Confidence should be lower for partial data
        assert confidence < 1.0

    @pytest.mark.asyncio
    async def test_calculate_confidence(self, extractor: DataExtractor) -> None:
        """Test confidence calculation."""
        # Full data - high confidence
        full_data = MietvertragData(
            vermieter={"name": "Test", "adresse": "Test"},
            mieter={"name": "Test", "adresse": "Test"},
            objekt_adresse="Test",
            nettomiete_eur=1000,
            mietbeginn="2024-01-01",
        )
        confidence_full = extractor._calculate_confidence(full_data)

        # Empty data - low confidence
        empty_data = MietvertragData()
        confidence_empty = extractor._calculate_confidence(empty_data)

        assert confidence_full > confidence_empty
        assert 0 <= confidence_full <= 1
        assert 0 <= confidence_empty <= 1

    @pytest.mark.asyncio
    async def test_extract_raw(
        self, extractor: DataExtractor, mock_client: AsyncMock
    ) -> None:
        """Test extracting raw dictionary."""
        mock_data = {"nettomiete_eur": 1000}

        response = MagicMock()
        response.choices = [MagicMock(message=MagicMock(content=json.dumps(mock_data)))]
        mock_client.chat.completions.create = AsyncMock(return_value=response)

        result = await extractor.extract_raw("Text", DocumentType.MIETVERTRAG)

        assert isinstance(result, dict)
        assert "nettomiete_eur" in result
