from core.config import AppSettings
from core.use_cases import AnalyzeDatasetDriftUseCase
from data.repositories import CSVDatasetRepository
from drift.detectors import StatisticalDriftDetector


def build_drift_use_case(
    settings: AppSettings,
    reference_path: str | None = None,
    current_path: str | None = None,
) -> AnalyzeDatasetDriftUseCase:
    repository = CSVDatasetRepository(
        reference_path or settings.data.reference_dataset_path,
        current_path or settings.data.current_dataset_path,
    )
    detector = StatisticalDriftDetector(
        categorical_threshold=settings.thresholds.categorical_distance,
    )
    return AnalyzeDatasetDriftUseCase(
        repository=repository,
        detector=detector,
        numerical_threshold=settings.thresholds.numerical_p_value,
        min_rows=settings.monitoring.min_rows,
    )

