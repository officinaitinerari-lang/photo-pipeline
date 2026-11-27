"""Central configuration loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(
    {".jpg", ".jpeg", ".tif", ".tiff", ".png", ".nef", ".dng"}
)
RAW_EXTENSIONS: frozenset[str] = frozenset({".nef", ".dng"})


@dataclass(frozen=True)
class UploadTarget:
    """One microstock agency destination (FTPS or SFTP)."""

    name: str
    protocol: str  # "ftps" | "sftp"
    host: str
    port: int
    username: str
    password: str
    remote_dir: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=os.environ.get("PIPELINE_ENV_FILE", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Paths -----------------------------------------------------------------
    data_root: Path = Path("/data")
    fonts_dir: Path = Path("/app/assets/fonts")
    models_dir: Path = Path("/app/assets/models")

    # --- Watcher ---------------------------------------------------------------
    stable_seconds: float = 5.0
    watch_use_polling: bool = False
    poll_interval_seconds: float = 2.0
    analog_keywords: str = "analog,analogico,film,35mm,120mm,pellicola,kodak,ilford,fuji,portra,ektar,hp5,tri-x"
    ignore_dir_names: str = "@eaDir,#recycle,.@__thumb,@Recycle,.Trash,.Spotlight-V100,.ugreen_thumb"

    # --- Gemini ----------------------------------------------------------------
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_min_interval_seconds: float = 6.0
    gemini_max_retries: int = 6
    gemini_backoff_base_seconds: float = 4.0
    gemini_backoff_max_seconds: float = 120.0
    gemini_timeout_seconds: int = 120
    gemini_thinking_budget: int = -1
    gemini_preview_long_side: int = 1024
    gemini_quota_cooldown_minutes: int = 60

    # --- Image processing --------------------------------------------------------
    upscale_threshold_px: int = 6000
    upscale_target_long_side: int = 7200
    upscale_engine: str = "onnx"  # "onnx" | "lanczos"
    upscale_model_file: str = "realesr-general-x4v3.onnx"
    upscale_tile_size: int = 512
    upscale_tile_overlap: int = 24
    onnx_threads: int = 2
    print_dpi: int = 300
    max_print_file_mb: float = 18.5
    jpeg_quality_prints: int = 95
    jpeg_quality_min: int = 82

    # --- Etsy packaging ------------------------------------------------------------
    etsy_zip_max_mb: float = 19.0  # Etsy caps each file at 20 MB (decimal): 19 MiB = 19.9 MB
    etsy_max_files: int = 5  # Etsy accepts at most 5 digital files per listing
    jpeg_quality_step: int = 4  # quality drop per retry when the prints don't fit in etsy_max_files

    # --- Microstock ------------------------------------------------------------------
    microstock_jpeg_quality: int = 95
    microstock_min_megapixels: float = 4.0
    microstock_keep_gps: bool = False
    author_name: str = ""
    copyright_notice: str = ""

    # --- Upload ------------------------------------------------------------------------
    dry_run: bool = True
    upload_max_retries: int = 3
    upload_timeout_seconds: int = 120
    sftp_known_hosts: str = ""  # default: <state>/known_hosts

    adobe_enabled: bool = False
    adobe_protocol: str = "sftp"
    adobe_host: str = "sftp.contributor.adobestock.com"
    adobe_port: int = 22
    adobe_username: str = ""
    adobe_password: str = ""
    adobe_remote_dir: str = "/"

    shutterstock_enabled: bool = False
    shutterstock_protocol: str = "ftps"
    shutterstock_host: str = "ftps.shutterstock.com"
    shutterstock_port: int = 21
    shutterstock_username: str = ""
    shutterstock_password: str = ""
    shutterstock_remote_dir: str = "/"

    # --- Etsy drafts (Open API v3; drafts only, the seller publishes by hand) -----------
    etsy_enabled: bool = False
    etsy_api_key: str = ""  # app keystring
    etsy_shared_secret: str = ""
    etsy_redirect_uri: str = ""  # https callback registered on etsy.com/developers
    etsy_language: str = "it"  # the shop's main language: "it" or "en"
    etsy_translate: bool = True  # also add the other language (it <-> en) as an Etsy translation
    etsy_price: float = 4.90
    etsy_quantity: int = 999
    etsy_taxonomy_id: int = 0  # 0 = look up "Digital Prints" automatically
    etsy_when_made: str = ""  # "" = current era, e.g. 2020_2026
    etsy_sync_minutes: int = 5

    # --- Notifications -------------------------------------------------------------------
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # --- Worker ----------------------------------------------------------------------------
    max_job_attempts: int = 3
    retry_delay_minutes: int = 10
    worker_idle_seconds: float = 2.0
    log_level: str = "INFO"
    log_format: str = "json"  # "json" | "text"

    @field_validator("upscale_engine", "adobe_protocol", "shutterstock_protocol", "log_format")
    @classmethod
    def _lower(cls, v: str) -> str:
        return v.strip().lower()

    @field_validator("gemini_min_interval_seconds", "stable_seconds")
    @classmethod
    def _non_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("must be >= 0")
        return v

    # --- Derived paths -----------------------------------------------------------------------
    @property
    def input_dir(self) -> Path:
        return self.data_root / "input"

    @property
    def etsy_dir(self) -> Path:
        return self.data_root / "output" / "etsy"

    @property
    def microstock_uploaded_dir(self) -> Path:
        return self.data_root / "output" / "microstock_uploaded"

    @property
    def microstock_pending_dir(self) -> Path:
        return self.data_root / "output" / "microstock_pending"

    @property
    def microstock_review_dir(self) -> Path:
        return self.data_root / "output" / "microstock_review_editorial"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def failed_dir(self) -> Path:
        return self.data_root / "failed"

    @property
    def state_dir(self) -> Path:
        return self.data_root / "state"

    @property
    def work_dir(self) -> Path:
        return self.state_dir / "work"

    @property
    def db_path(self) -> Path:
        return self.state_dir / "pipeline.db"

    @property
    def heartbeat_path(self) -> Path:
        return self.state_dir / "heartbeat"

    @property
    def known_hosts_path(self) -> Path:
        return Path(self.sftp_known_hosts) if self.sftp_known_hosts else self.state_dir / "known_hosts"

    @property
    def upscale_model_path(self) -> Path:
        return self.models_dir / self.upscale_model_file

    @property
    def analog_keyword_list(self) -> list[str]:
        return [k.strip().lower() for k in self.analog_keywords.split(",") if k.strip()]

    @property
    def ignore_dir_set(self) -> set[str]:
        return {d.strip() for d in self.ignore_dir_names.split(",") if d.strip()}

    @property
    def etsy_zip_max_bytes(self) -> int:
        return int(self.etsy_zip_max_mb * 1024 * 1024)

    @property
    def max_print_file_bytes(self) -> int:
        return int(self.max_print_file_mb * 1024 * 1024)

    @property
    def telegram_enabled(self) -> bool:
        return bool(self.telegram_bot_token and self.telegram_chat_id)

    def upload_targets(self) -> list[UploadTarget]:
        targets: list[UploadTarget] = []
        for name in ("adobe", "shutterstock"):
            if not getattr(self, f"{name}_enabled"):
                continue
            protocol = getattr(self, f"{name}_protocol")
            if protocol not in {"ftps", "sftp"}:
                raise ValueError(f"{name.upper()}_PROTOCOL must be 'ftps' or 'sftp', got {protocol!r}")
            targets.append(
                UploadTarget(
                    name=name,
                    protocol=protocol,
                    host=getattr(self, f"{name}_host"),
                    port=int(getattr(self, f"{name}_port")),
                    username=getattr(self, f"{name}_username"),
                    password=getattr(self, f"{name}_password"),
                    remote_dir=getattr(self, f"{name}_remote_dir") or "/",
                )
            )
        return targets

    def ensure_directories(self) -> None:
        for d in (
            self.input_dir,
            self.etsy_dir,
            self.microstock_uploaded_dir,
            self.microstock_pending_dir,
            self.microstock_review_dir,
            self.processed_dir,
            self.failed_dir,
            self.state_dir,
            self.work_dir,
        ):
            d.mkdir(parents=True, exist_ok=True)

    def validate_runtime(self) -> list[str]:
        """Return a list of blocking configuration problems (empty = OK)."""
        problems: list[str] = []
        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is empty: the AI analysis step cannot run.")
        try:
            for t in self.upload_targets():
                if not (t.username and t.password):
                    problems.append(f"Upload target '{t.name}' enabled but username/password missing.")
        except ValueError as exc:
            problems.append(str(exc))
        if self.etsy_enabled:
            if not (self.etsy_api_key and self.etsy_shared_secret):
                problems.append("ETSY_ENABLED is true but ETSY_API_KEY / ETSY_SHARED_SECRET are missing.")
            if not self.etsy_redirect_uri.startswith("https://"):
                problems.append("ETSY_REDIRECT_URI must be the https callback registered in the Etsy app.")
        return problems


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
