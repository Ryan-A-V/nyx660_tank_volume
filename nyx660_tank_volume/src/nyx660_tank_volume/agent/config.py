"""Pydantic configuration models for the nyx660-agent process."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentIdentity(BaseModel):
    """OPS resolves the unit from the token. unit_id is informational only,
    used in agent-side logging."""

    unit_id: str = Field(default="unknown")
    version: str = Field(default="0.1.0")


class OpsConfig(BaseModel):
    base_url: str = Field(min_length=1)
    api_token: Optional[str] = None
    api_token_env: Optional[str] = None

    request_timeout_s: float = Field(default=30.0, gt=0)
    connect_timeout_s: float = Field(default=10.0, gt=0)
    backoff_initial_s: float = Field(default=5.0, gt=0)
    backoff_max_s: float = Field(default=300.0, gt=0)

    @model_validator(mode="after")
    def _resolve_token(self) -> "OpsConfig":
        if self.api_token_env:
            env_value = os.environ.get(self.api_token_env)
            if not env_value:
                raise ValueError(
                    f"ops.api_token_env={self.api_token_env!r} is set but the "
                    f"environment variable is empty"
                )
            self.api_token = env_value
        if not self.api_token:
            raise ValueError("ops.api_token must be set (directly or via api_token_env)")
        return self

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class LocalApiConfig(BaseModel):
    base_url: str = "http://127.0.0.1:8080"
    api_token: str = Field(min_length=1)
    request_timeout_s: float = Field(default=15.0, gt=0)

    @field_validator("base_url")
    @classmethod
    def _strip_trailing_slash(cls, v: str) -> str:
        return v.rstrip("/")


class PollConfig(BaseModel):
    interval_s: float = Field(default=30.0, gt=0)
    attach_latest_measurement: bool = True
    command_timeout_s: float = Field(default=120.0, gt=0)


class StorageConfig(BaseModel):
    db_path: Path = Path("./data/agent.db")


class ConfigPushConfig(BaseModel):
    target_config_path: Path = Path("./config.yaml")
    backup_config_path: Path = Path("./config.yaml.prev")
    service_name: str = "helios2-tank-volume.service"
    health_check_timeout_s: float = Field(default=60.0, gt=0)
    health_check_interval_s: float = Field(default=2.0, gt=0)


class AlertDefaults(BaseModel):
    """Fallback alert thresholds used until OPS pushes a config with operator-set
    values. The agent caches values from the most recent OPS config and falls
    back to these when a field is missing or no config has been pushed yet."""

    tank_full_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    tank_low_threshold: float = Field(default=0.05, ge=0.0, le=1.0)
    min_valid_pixel_ratio: float = Field(default=0.30, ge=0.0, le=1.0)
    calibration_max_age_seconds: float = Field(default=30 * 24 * 3600, gt=0)


class AgentConfig(BaseModel):
    """Top-level agent configuration, loaded from agent.yaml."""

    agent: AgentIdentity = AgentIdentity()
    ops: OpsConfig
    local: LocalApiConfig
    poll: PollConfig = PollConfig()
    storage: StorageConfig = StorageConfig()
    config_push: ConfigPushConfig = ConfigPushConfig()
    alert_defaults: AlertDefaults = AlertDefaults()


def load_agent_config(path: Path) -> AgentConfig:
    """Load and validate agent configuration from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Agent config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AgentConfig.model_validate(raw)
