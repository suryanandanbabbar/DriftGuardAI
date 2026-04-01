from pydantic import BaseModel, Field


class AnalyzeDriftRequest(BaseModel):
    dataset_name: str = "production"
    reference_path: str | None = None
    current_path: str | None = None
    use_predefined_paths: bool = True


class DriftMetricResponse(BaseModel):
    metric_name: str
    value: float | None
    threshold: float
    drift_detected: bool
    p_value: float | None = None
    interpretation: str | None = None


class FeatureDriftMetricsResponse(BaseModel):
    psi: DriftMetricResponse | None = None
    ks: DriftMetricResponse | None = None
    kl_divergence: DriftMetricResponse | None = None
    chi_square: DriftMetricResponse | None = None
    distribution_difference: DriftMetricResponse | None = None


class FeatureDriftResponse(BaseModel):
    feature_name: str
    feature_type: str
    reference_size: int
    current_size: int
    drift_detected: bool
    supported: bool
    reason: str | None = None
    metrics: FeatureDriftMetricsResponse


class DriftDetectionResponse(BaseModel):
    dataset_name: str
    generated_at: str
    total_features: int
    drifted_features_count: int
    stable_features_count: int
    features: list[FeatureDriftResponse]


class ErrorResponse(BaseModel):
    detail: str
