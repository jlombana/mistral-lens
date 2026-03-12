"""Configuration module using pydantic-settings.

Loads settings from environment variables and .env file.
Single source of configuration for the entire application.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Mistral API
    MISTRAL_API_KEY: str

    # Model configuration
    OCR_MODEL: str = "mistral-ocr-latest"
    CHAT_MODEL: str = "mistral-large-latest"

    # Paths
    DATASET_PATH: str = "data/"
    RESULTS_PATH: str = "results/"

    # Gradio server
    HOST: str = "0.0.0.0"
    PORT: int = 7860

    @property
    def dataset_dir(self) -> Path:
        """Return dataset path as a Path object."""
        return Path(self.DATASET_PATH)

    @property
    def results_dir(self) -> Path:
        """Return results path as a Path object."""
        return Path(self.RESULTS_PATH)


@lru_cache
def get_settings() -> Settings:
    """Return cached Settings singleton.

    Returns:
        Settings instance loaded from environment.
    """
    return Settings()
