from functools import lru_cache
from pathlib import Path
import os
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.yaml"


class APISettings(BaseModel):
    title: str = "DriftGuardAI API"
    version: str = "0.1.0"
    prefix: str = "/api/v1"


class ThresholdSettings(BaseModel):
    numerical_p_value: float = 0.05
    categorical_distance: float = 0.10
    missing_rate_delta: float = 0.02
    psi: float = 0.20
    ks_significance_level: float = 0.05
    kl_divergence: float = 0.10
    histogram_bins: int = 10
    histogram_strategy: Literal["quantile", "uniform"] = "quantile"
    histogram_epsilon: float = 1e-6


class MonitoringSettings(BaseModel):
    min_rows: int = 100
    default_method: str = "kolmogorov_smirnov"
    alert_on_missing_data: bool = True


class DataSettings(BaseModel):
    reference_dataset_path: str = "datasets/reference.csv"
    current_dataset_path: str = "datasets/current.csv"


class AppSettings(BaseModel):
    environment: str = "development"
    debug: bool = True
    log_level: str = "INFO"
    api: APISettings = Field(default_factory=APISettings)
    thresholds: ThresholdSettings = Field(default_factory=ThresholdSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    data: DataSettings = Field(default_factory=DataSettings)


def _load_yaml_file(config_path: Path) -> dict:
    if not config_path.exists():
        return {}

    with config_path.open("r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file) or {}


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    load_dotenv()
    configured_path = os.getenv("DRIFT_GUARD_CONFIG_PATH")
    config_path = Path(configured_path) if configured_path else DEFAULT_CONFIG_PATH
    raw_config = _load_yaml_file(config_path)
    return AppSettings(**raw_config)
