# -------------------------------------------------------------- #
#   Centralized Configuration — Pydantic Settings                  #
#   Dave Pattern: COPY — Pydantic Settings, never os.environ       #
# -------------------------------------------------------------- #

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from .env file.
    
    All configuration flows through this single source of truth.
    Never use os.environ or os.getenv directly — always use settings.
    """

    # Google Cloud
    gcp_project: str
    gcp_location: str = "us-central1"

    # Model Configuration
    gemini_pro_model: str = "gemini-2.5-pro"
    gemini_flash_model: str = "gemini-2.5-flash"
    embedding_model: str = "text-embedding-004"

    # App Settings
    app_env: str = "development"
    log_level: str = "INFO"
    max_conversation_history: int = 20
    genome_dir: str = "profiles"
    lancedb_path: str = "data/lancedb"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


# Module-level singleton — import this everywhere
settings = Settings()
