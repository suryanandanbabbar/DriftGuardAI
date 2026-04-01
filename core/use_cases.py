from datetime import datetime, timezone

from core.config import get_settings
from core.entities import DriftAnalysisReport
from core.exceptions import DataValidationError
from core.interfaces import DatasetRepository, DriftDetector


class AnalyzeDatasetDriftUseCase:
    def __init__(
        self,
        repository: DatasetRepository,
        detector: DriftDetector,
        numerical_threshold: float,
        min_rows: int,
    ) -> None:
        self.repository = repository
        self.detector = detector
        self.numerical_threshold = numerical_threshold
        self.min_rows = min_rows

    def execute(
        self,
        dataset_name: str | None = None,
        columns: list[str] | None = None,
    ) -> DriftAnalysisReport:
        resolved_dataset_name = dataset_name or get_settings().runtime.default_dataset_name
        reference_dataset, current_dataset = self.repository.load_datasets()

        if len(reference_dataset) < self.min_rows or len(current_dataset) < self.min_rows:
            raise DataValidationError(
                "Both datasets must contain at least "
                f"{self.min_rows} rows before running drift analysis.",
            )

        available_columns = [
            column_name
            for column_name in reference_dataset.columns
            if column_name in current_dataset.columns
        ]

        if columns:
            missing_columns = sorted(set(columns) - set(available_columns))
            if missing_columns:
                raise DataValidationError(
                    f"Requested columns are unavailable in both datasets: {missing_columns}",
                )
            target_columns = columns
        else:
            target_columns = available_columns

        if not target_columns:
            raise DataValidationError("No shared columns found between reference and current datasets.")

        report = DriftAnalysisReport(
            dataset_name=resolved_dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
        )

        for column_name in target_columns:
            result = self.detector.analyze(
                column_name=column_name,
                reference=reference_dataset[column_name],
                current=current_dataset[column_name],
                threshold=self.numerical_threshold,
            )
            if result.drift_detected:
                report.drifted_columns.append(result)
            else:
                report.stable_columns.append(result)

        return report
