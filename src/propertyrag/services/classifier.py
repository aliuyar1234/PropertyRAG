"""Document type classification service."""

from openai import AsyncOpenAI

from propertyrag.core.config import get_settings
from propertyrag.core.logging import get_logger
from propertyrag.core.models import DocumentType

logger = get_logger(__name__)

CLASSIFICATION_PROMPT = """Analysiere den folgenden Dokumenttext und bestimme den Dokumenttyp.

Mögliche Dokumenttypen:
- mietvertrag: Mietverträge, Pachtverträge, Nutzungsverträge für Immobilien
- gutachten: Verkehrswertgutachten, Immobilienbewertungen, Sachverständigengutachten
- grundbuchauszug: Grundbuchauszüge, Grundbuchblätter, Eigentumsauskünfte
- nebenkostenabrechnung: Betriebskostenabrechnungen, Nebenkostenabrechnungen, Hausgeldabrechnungen
- unknown: Falls keiner der obigen Typen passt

Antworte NUR mit einem der folgenden Wörter (kleingeschrieben):
mietvertrag, gutachten, grundbuchauszug, nebenkostenabrechnung, unknown

Dokumenttext (erste 3000 Zeichen):
{text}

Dokumenttyp:"""


class ClassificationError(Exception):
    """Error during document classification."""

    pass


class DocumentClassifier:
    """Service for classifying document types using LLM."""

    def __init__(self, client: AsyncOpenAI | None = None) -> None:
        """
        Initialize the classifier.

        Args:
            client: Optional AsyncOpenAI client. If not provided, creates one.
        """
        settings = get_settings()
        self.client = client or AsyncOpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_chat_model

        logger.info("classifier_initialized", model=self.model)

    async def classify(self, text: str) -> DocumentType:
        """
        Classify a document based on its text content.

        Args:
            text: Document text to classify.

        Returns:
            Detected DocumentType.

        Raises:
            ClassificationError: If classification fails.
        """
        if not text.strip():
            logger.warning("empty_text_for_classification")
            return DocumentType.UNKNOWN

        # Use first 3000 characters for classification
        sample_text = text[:3000]

        logger.info("classifying_document", text_length=len(text))

        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": CLASSIFICATION_PROMPT.format(text=sample_text),
                    }
                ],
                max_tokens=20,
                temperature=0,
            )

            result = response.choices[0].message.content.strip().lower()

            # Map result to DocumentType
            type_mapping = {
                "mietvertrag": DocumentType.MIETVERTRAG,
                "gutachten": DocumentType.GUTACHTEN,
                "grundbuchauszug": DocumentType.GRUNDBUCHAUSZUG,
                "nebenkostenabrechnung": DocumentType.NEBENKOSTENABRECHNUNG,
                "unknown": DocumentType.UNKNOWN,
            }

            document_type = type_mapping.get(result, DocumentType.UNKNOWN)

            logger.info(
                "document_classified",
                result=result,
                document_type=document_type.value,
            )

            return document_type

        except Exception as e:
            logger.error("classification_error", error=str(e))
            raise ClassificationError(f"Failed to classify document: {e}") from e
