from __future__ import annotations

from io import BytesIO

import pandas as pd
from fastapi import UploadFile

from core.config import AppSettings
from core.exceptions import DataValidationError
from data.ingestion import CSVDataIngestion
from drift.detectors import DriftDetector
from utils.dataset_validation import validate_compatible_datasets, validate_non_empty_dataset


def build_detector_from_paths(
    settings: AppSettings,
    reference_path: str | None = None,
    current_path: str | None = None,
    use_predefined_paths: bool = True,
) -> DriftDetector:
    resolved_reference_path, resolved_current_path = resolve_dataset_paths(
        settings=settings,
        reference_path=reference_path,
        current_path=current_path,
        use_predefined_paths=use_predefined_paths,
    )
    ingestion = CSVDataIngestion(resolved_reference_path, resolved_current_path)
    baseline_dataset, incoming_dataset = ingestion.load_datasets()
    return DriftDetector(
        baseline_dataset=baseline_dataset,
        incoming_dataset=incoming_dataset,
        thresholds=settings.thresholds,
    )


async def build_detector_from_uploads(
    settings: AppSettings,
    reference_file: UploadFile,
    current_file: UploadFile,
) -> DriftDetector:
    baseline_dataset = await _read_uploaded_csv(reference_file, dataset_name="Baseline")
    incoming_dataset = await _read_uploaded_csv(current_file, dataset_name="Incoming")
    validate_non_empty_dataset(baseline_dataset, "Baseline")
    validate_non_empty_dataset(incoming_dataset, "Incoming")
    validate_compatible_datasets(baseline_dataset, incoming_dataset)
    return DriftDetector(
        baseline_dataset=baseline_dataset,
        incoming_dataset=incoming_dataset,
        thresholds=settings.thresholds,
    )


def resolve_dataset_paths(
    settings: AppSettings,
    reference_path: str | None,
    current_path: str | None,
    use_predefined_paths: bool,
) -> tuple[str, str]:
    if reference_path and current_path:
        return reference_path, current_path

    if reference_path or current_path:
        raise DataValidationError("Both reference_path and current_path must be provided together.")

    if not use_predefined_paths:
        raise DataValidationError(
            "Provide both dataset paths or enable use_predefined_paths to use config.yaml defaults.",
        )

    return settings.data.reference_dataset_path, settings.data.current_dataset_path


async def _read_uploaded_csv(upload: UploadFile, dataset_name: str) -> pd.DataFrame:
    if not upload.filename:
        raise DataValidationError(f"{dataset_name} upload must include a filename.")
    if not upload.filename.lower().endswith(".csv"):
        raise DataValidationError(
            f"{dataset_name} upload must be a CSV file. Received '{upload.filename}'.",
        )

    raw_bytes = await upload.read()
    if not raw_bytes:
        raise DataValidationError(f"{dataset_name} upload is empty.")

    try:
        dataframe = pd.read_csv(BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - defensive API boundary
        raise DataValidationError(
            f"Failed to parse {dataset_name.lower()} CSV upload '{upload.filename}': {exc}",
        ) from exc
    finally:
        await upload.close()

    return dataframe
