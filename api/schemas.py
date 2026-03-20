from pydantic import BaseModel, Field


class AnalyzeDriftRequest(BaseModel):
    dataset_name: str = "production"
    reference_path: str | None = None
    current_path: str | None = None
    columns: list[str] = Field(default_factory=list)


class ColumnDriftResponse(BaseModel):
    column_name: str
    method: str
    statistic: float
    p_value: float
    drift_detected: bool
    reference_size: int
    current_size: int


class DriftAnalysisResponse(BaseModel):
    dataset_name: str
    generated_at: str
    total_columns: int
    drifted_columns: list[ColumnDriftResponse]
    stable_columns: list[ColumnDriftResponse]

