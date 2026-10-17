from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="FRESHNET_",
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "FreshNet API"
    app_version: str = "0.1.0"
    environment: str = "development"
    log_level: str = "INFO"
    host: str = "127.0.0.1"
    port: int = 8000
    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"
    sqlite_path: str = "data/local/freshnet.sqlite"
    checkpoint_path: str = "artifacts/checkpoints/freshnet.pt"
    bundle_dir: str = "artifacts/deployment/v3/fused_epoch29_b962c650ed80"
    # Separate produce-type bundle (empty string disables automatic identification).
    produce_bundle_dir: str = "artifacts/deployment/produce_v1/effb0_epoch10_e389c317c8e5"
    device: str = "cpu"

    @property
    def bundle_directory(self) -> Path:
        return self.resolve_path(self.bundle_dir)

    @property
    def produce_bundle_directory(self) -> Path:
        return self.resolve_path(self.produce_bundle_dir)
    config_path: str = "configs/default.yaml"
    architecture_name: str = "FusedFreshNet"

    @property
    def cors_origin_list(self) -> list[str]:
        return [item.strip() for item in self.cors_origins.split(",") if item.strip()]

    def resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return (REPO_ROOT / path).resolve()

    @property
    def sqlite_file(self) -> Path:
        return self.resolve_path(self.sqlite_path)

    @property
    def checkpoint_file(self) -> Path:
        return self.resolve_path(self.checkpoint_path)


@lru_cache
def get_settings() -> Settings:
    return Settings()
