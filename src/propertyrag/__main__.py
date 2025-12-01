"""Main entry point for running the application."""

import uvicorn

from propertyrag.core.config import get_settings


def main() -> None:
    """Run the application."""
    settings = get_settings()

    uvicorn.run(
        "propertyrag.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )


if __name__ == "__main__":
    main()
