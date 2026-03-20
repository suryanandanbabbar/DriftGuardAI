from dataclasses import dataclass, field


@dataclass(slots=True)
class ColumnDriftResult:
    column_name: str
    method: str
    statistic: float
    p_value: float
    drift_detected: bool
    reference_size: int
    current_size: int


@dataclass(slots=True)
class DriftAnalysisReport:
    dataset_name: str
    generated_at: str
    drifted_columns: list[ColumnDriftResult] = field(default_factory=list)
    stable_columns: list[ColumnDriftResult] = field(default_factory=list)

    @property
    def total_columns(self) -> int:
        return len(self.drifted_columns) + len(self.stable_columns)

