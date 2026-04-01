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


@dataclass(slots=True)
class DriftMetricResult:
    metric_name: str
    value: float | None
    threshold: float
    drift_detected: bool
    p_value: float | None = None
    interpretation: str | None = None


@dataclass(slots=True)
class FeatureDriftMetrics:
    psi: DriftMetricResult | None = None
    ks: DriftMetricResult | None = None
    kl_divergence: DriftMetricResult | None = None
    chi_square: DriftMetricResult | None = None
    distribution_difference: DriftMetricResult | None = None


@dataclass(slots=True)
class FeatureDriftResult:
    feature_name: str
    feature_type: str
    reference_size: int
    current_size: int
    drift_detected: bool
    supported: bool = True
    metrics: FeatureDriftMetrics = field(default_factory=FeatureDriftMetrics)
    reason: str | None = None


@dataclass(slots=True)
class FeatureDriftReport:
    dataset_name: str
    generated_at: str
    features: list[FeatureDriftResult] = field(default_factory=list)

    @property
    def total_features(self) -> int:
        return len(self.features)

    @property
    def drifted_features(self) -> list[FeatureDriftResult]:
        return [feature for feature in self.features if feature.drift_detected]

    @property
    def stable_features(self) -> list[FeatureDriftResult]:
        return [feature for feature in self.features if not feature.drift_detected]
