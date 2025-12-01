"""Structured data extraction service using OpenAI."""

import json
from typing import Any

from openai import AsyncOpenAI
from pydantic import ValidationError

from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger
from propertyrag.core.models import (
    DocumentType,
    ExtractedData,
    GrundbuchauszugData,
    GutachtenData,
    MietvertragData,
    NebenkostenabrechnungData,
)
from propertyrag.services.extraction_prompts import EXTRACTION_PROMPTS

logger = get_logger(__name__)

# Mapping of document types to Pydantic models
EXTRACTION_MODELS: dict[DocumentType, type[ExtractedData]] = {
    DocumentType.MIETVERTRAG: MietvertragData,
    DocumentType.GUTACHTEN: GutachtenData,
    DocumentType.GRUNDBUCHAUSZUG: GrundbuchauszugData,
    DocumentType.NEBENKOSTENABRECHNUNG: NebenkostenabrechnungData,
}


class ExtractionError(Exception):
    """Error during data extraction."""

    pass


class DataExtractor:
    """Service for extracting structured data from documents."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        """
        Initialize the extractor.

        Args:
            client: Optional AsyncOpenAI client. If not provided, creates one.
        """
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

        logger.info("extractor_initialized", model=self.model)

    async def extract(
        self, text: str, document_type: DocumentType
    ) -> tuple[ExtractedData, float]:
        """
        Extract structured data from document text.

        Args:
            text: Full document text.
            document_type: Type of document to extract.

        Returns:
            Tuple of (extracted data model, confidence score).

        Raises:
            ExtractionError: If extraction fails or document type not supported.
        """
        if document_type == DocumentType.UNKNOWN:
            raise ExtractionError("Cannot extract data from unknown document type")

        if document_type not in EXTRACTION_MODELS:
            raise ExtractionError(f"Unsupported document type: {document_type}")

        prompt_template = EXTRACTION_PROMPTS.get(document_type.value)
        if not prompt_template:
            raise ExtractionError(f"No extraction prompt for: {document_type}")

        logger.info(
            "extracting_data",
            document_type=document_type.value,
            text_length=len(text),
        )

        try:
            # Use response_format for JSON output
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "Du bist ein Experte für die Analyse von Immobiliendokumenten. "
                        "Extrahiere die angeforderten Informationen präzise und vollständig. "
                        "Antworte ausschließlich im JSON-Format.",
                    },
                    {
                        "role": "user",
                        "content": prompt_template.format(text=text),
                    },
                ],
                response_format={"type": "json_object"},
                temperature=0,
            )

            raw_json = response.choices[0].message.content

            # Parse JSON response
            try:
                data_dict = json.loads(raw_json)
            except json.JSONDecodeError as e:
                logger.error("json_parse_error", error=str(e), raw=raw_json[:500])
                raise ExtractionError(f"Failed to parse JSON response: {e}") from e

            # Validate and convert to Pydantic model
            model_class = EXTRACTION_MODELS[document_type]
            try:
                extracted_data = model_class.model_validate(data_dict)
            except ValidationError as e:
                logger.warning(
                    "validation_error",
                    error=str(e),
                    data=data_dict,
                )
                # Try to salvage what we can by cleaning the data
                cleaned_data = self._clean_data(data_dict, model_class)
                extracted_data = model_class.model_validate(cleaned_data)

            # Calculate confidence based on how many fields were extracted
            confidence = self._calculate_confidence(extracted_data)

            logger.info(
                "extraction_completed",
                document_type=document_type.value,
                confidence=confidence,
            )

            return extracted_data, confidence

        except ExtractionError:
            raise
        except Exception as e:
            logger.error("extraction_error", error=str(e))
            raise ExtractionError(f"Extraction failed: {e}") from e

    def _clean_data(self, data: dict[str, Any], model_class: type) -> dict[str, Any]:
        """Clean and normalize extracted data to match model expectations."""
        cleaned = {}

        # Get model fields
        model_fields = model_class.model_fields

        for field_name, field_info in model_fields.items():
            if field_name not in data:
                continue

            value = data[field_name]

            # Skip None values
            if value is None:
                cleaned[field_name] = None
                continue

            # Handle numeric fields that might come as strings
            field_type = field_info.annotation
            if field_type in (int, float, "int | None", "float | None"):
                if isinstance(value, str):
                    # Try to extract number from string
                    value = self._extract_number(value)

            cleaned[field_name] = value

        return cleaned

    def _extract_number(self, value: str) -> float | int | None:
        """Extract a number from a string."""
        import re

        if not value:
            return None

        # Remove common formatting
        cleaned = value.replace(".", "").replace(",", ".").replace(" ", "")
        cleaned = re.sub(r"[€$%]", "", cleaned)

        # Try to find a number
        match = re.search(r"-?\d+\.?\d*", cleaned)
        if match:
            num_str = match.group()
            if "." in num_str:
                return float(num_str)
            return int(num_str)

        return None

    def _calculate_confidence(self, data: ExtractedData) -> float:
        """
        Calculate extraction confidence based on filled fields.

        Returns a score between 0 and 1.
        """
        fields = data.model_fields
        total_fields = len(fields)
        filled_fields = 0

        for field_name in fields:
            value = getattr(data, field_name, None)
            if value is not None:
                if isinstance(value, list) and len(value) == 0:
                    continue
                filled_fields += 1

        return filled_fields / total_fields if total_fields > 0 else 0.0

    async def extract_raw(
        self, text: str, document_type: DocumentType
    ) -> dict[str, Any]:
        """
        Extract data and return as raw dictionary.

        Useful when you need the raw data without Pydantic validation.

        Args:
            text: Full document text.
            document_type: Type of document to extract.

        Returns:
            Extracted data as dictionary.
        """
        data, _ = await self.extract(text, document_type)
        return data.model_dump()
