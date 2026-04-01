from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from core.exceptions import DataValidationError, DatasetFileNotFoundError
from utils.dataset_validation import validate_compatible_datasets, validate_non_empty_dataset
from utils.logging import get_logger, log_event

logger = get_logger(__name__)


class CSVDataIngestion:
    def __init__(self, baseline_path: str | Path, incoming_path: str | Path) -> None:
        self.baseline_path = self._resolve_path(baseline_path)
        self.incoming_path = self._resolve_path(incoming_path)
        log_event(
            logger,
            logging.INFO,
            "Initialized CSV data ingestion.",
            event="data_ingestion_initialized",
            baseline_path=str(self.baseline_path),
            incoming_path=str(self.incoming_path),
        )

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        log_event(
            logger,
            logging.INFO,
            "Loading baseline and incoming datasets.",
            event="datasets_loading_started",
            baseline_path=str(self.baseline_path),
            incoming_path=str(self.incoming_path),
        )
        baseline_dataset = self._load_csv(self.baseline_path, dataset_name="Baseline")
        incoming_dataset = self._load_csv(self.incoming_path, dataset_name="Incoming")
        validate_compatible_datasets(baseline_dataset, incoming_dataset)
        log_event(
            logger,
            logging.INFO,
            "Successfully loaded and validated datasets.",
            event="datasets_loading_completed",
            baseline_rows=int(baseline_dataset.shape[0]),
            incoming_rows=int(incoming_dataset.shape[0]),
            columns=list(baseline_dataset.columns),
        )
        return baseline_dataset, incoming_dataset

    def load_baseline_dataset(self) -> pd.DataFrame:
        dataset = self._load_csv(self.baseline_path, dataset_name="Baseline")
        validate_non_empty_dataset(dataset, "Baseline")
        return dataset

    def load_incoming_dataset(self) -> pd.DataFrame:
        dataset = self._load_csv(self.incoming_path, dataset_name="Incoming")
        validate_non_empty_dataset(dataset, "Incoming")
        return dataset

    def _load_csv(self, path: Path, dataset_name: str) -> pd.DataFrame:
        if not path.exists():
            log_event(
                logger,
                logging.ERROR,
                "Dataset file not found.",
                event="dataset_file_missing",
                dataset_name=dataset_name,
                path=str(path),
            )
            raise DatasetFileNotFoundError(f"{dataset_name} dataset file not found: {path}")

        try:
            log_event(
                logger,
                logging.INFO,
                "Reading dataset from CSV.",
                event="dataset_read_started",
                dataset_name=dataset_name,
                path=str(path),
            )
            dataset = pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            log_event(
                logger,
                logging.ERROR,
                "Failed to read dataset from CSV.",
                event="dataset_read_failed",
                dataset_name=dataset_name,
                path=str(path),
                error=str(exc),
            )
            raise DataValidationError(
                f"Failed to load {dataset_name.lower()} dataset from {path}: {exc}",
            ) from exc

        log_event(
            logger,
            logging.INFO,
            "Read dataset from CSV.",
            event="dataset_read_completed",
            dataset_name=dataset_name,
            path=str(path),
            rows=int(dataset.shape[0]),
            columns=list(dataset.columns),
        )
        return dataset

    @staticmethod
    def _resolve_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return Path.cwd() / path
