from datetime import datetime, timezone

import numpy as np
import pandas as pd

from core.config import ThresholdSettings, get_settings
from core.entities import (
    ColumnDriftResult,
    DriftMetricResult,
    FeatureDriftMetrics,
    FeatureDriftReport,
    FeatureDriftResult,
)
from core.interfaces import DriftDetector as BaseDriftDetector
from drift.metrics import (
    kolmogorov_smirnov_test,
    kullback_leibler_divergence,
    population_stability_index,
)
from utils.dataset_validation import validate_compatible_datasets


class DriftDetector:
    def __init__(
        self,
        baseline_dataset: pd.DataFrame,
        incoming_dataset: pd.DataFrame,
        thresholds: ThresholdSettings | None = None,
    ) -> None:
        self.baseline_dataset = baseline_dataset.copy()
        self.incoming_dataset = incoming_dataset.copy()
        self.thresholds = thresholds or get_settings().thresholds
        validate_compatible_datasets(self.baseline_dataset, self.incoming_dataset)

    def generate_report(self, dataset_name: str = "incoming") -> FeatureDriftReport:
        features: list[FeatureDriftResult] = []

        for feature_name in self.baseline_dataset.columns:
            baseline_feature = self.baseline_dataset[feature_name]
            incoming_feature = self.incoming_dataset[feature_name]
            features.append(
                self._analyze_feature(
                    feature_name=feature_name,
                    baseline_feature=baseline_feature,
                    incoming_feature=incoming_feature,
                ),
            )

        return FeatureDriftReport(
            dataset_name=dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            features=features,
        )

    def _analyze_feature(
        self,
        feature_name: str,
        baseline_feature: pd.Series,
        incoming_feature: pd.Series,
    ) -> FeatureDriftResult:
        if not (
            pd.api.types.is_numeric_dtype(baseline_feature)
            and pd.api.types.is_numeric_dtype(incoming_feature)
        ):
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=str(baseline_feature.dtype),
                reference_size=int(baseline_feature.dropna().shape[0]),
                current_size=int(incoming_feature.dropna().shape[0]),
                drift_detected=False,
                supported=False,
                reason="PSI, KS, and KL divergence are only computed for numerical features.",
            )

        baseline_clean = baseline_feature.dropna()
        incoming_clean = incoming_feature.dropna()

        try:
            psi_value = population_stability_index(
                baseline_clean,
                incoming_clean,
                bins=self.thresholds.histogram_bins,
                strategy=self.thresholds.histogram_strategy,
                epsilon=self.thresholds.histogram_epsilon,
            )
            ks_result = kolmogorov_smirnov_test(
                baseline_clean,
                incoming_clean,
                significance_level=self.thresholds.ks_significance_level,
            )
            kl_value = kullback_leibler_divergence(
                baseline_clean,
                incoming_clean,
                bins=self.thresholds.histogram_bins,
                strategy=self.thresholds.histogram_strategy,
                epsilon=self.thresholds.histogram_epsilon,
            )
        except ValueError as exc:
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=str(baseline_feature.dtype),
                reference_size=int(baseline_clean.shape[0]),
                current_size=int(incoming_clean.shape[0]),
                drift_detected=False,
                supported=False,
                reason=str(exc),
            )

        psi_metric = DriftMetricResult(
            metric_name="psi",
            value=float(psi_value),
            threshold=float(self.thresholds.psi),
            drift_detected=bool(psi_value >= self.thresholds.psi),
            interpretation=(
                "Drift detected because PSI exceeded the configured threshold."
                if psi_value >= self.thresholds.psi
                else "No PSI-based drift detected."
            ),
        )
        ks_metric = DriftMetricResult(
            metric_name="kolmogorov_smirnov",
            value=float(ks_result.statistic),
            threshold=float(self.thresholds.ks_significance_level),
            drift_detected=ks_result.drift_detected,
            p_value=float(ks_result.p_value),
            interpretation=ks_result.interpretation,
        )
        kl_metric = DriftMetricResult(
            metric_name="kullback_leibler_divergence",
            value=float(kl_value),
            threshold=float(self.thresholds.kl_divergence),
            drift_detected=bool(kl_value >= self.thresholds.kl_divergence),
            interpretation=(
                "Drift detected because KL divergence exceeded the configured threshold."
                if kl_value >= self.thresholds.kl_divergence
                else "No KL-divergence-based drift detected."
            ),
        )

        return FeatureDriftResult(
            feature_name=feature_name,
            feature_type=str(baseline_feature.dtype),
            reference_size=int(baseline_clean.shape[0]),
            current_size=int(incoming_clean.shape[0]),
            drift_detected=any(
                metric.drift_detected for metric in (psi_metric, ks_metric, kl_metric)
            ),
            metrics=FeatureDriftMetrics(
                psi=psi_metric,
                ks=ks_metric,
                kl_divergence=kl_metric,
            ),
        )


class StatisticalDriftDetector(BaseDriftDetector):
    def __init__(self, categorical_threshold: float = 0.10) -> None:
        self.categorical_threshold = categorical_threshold

    def analyze(
        self,
        column_name: str,
        reference: pd.Series,
        current: pd.Series,
        threshold: float,
    ) -> ColumnDriftResult:
        reference_clean = reference.dropna()
        current_clean = current.dropna()

        if pd.api.types.is_numeric_dtype(reference_clean) and pd.api.types.is_numeric_dtype(current_clean):
            ks_result = kolmogorov_smirnov_test(
                reference_clean,
                current_clean,
                significance_level=threshold,
            )
            statistic = ks_result.statistic
            p_value = ks_result.p_value
            drift_detected = ks_result.drift_detected
            method = "kolmogorov_smirnov"
        else:
            statistic = self._categorical_distance(reference_clean, current_clean)
            p_value = 1.0
            drift_detected = statistic > self.categorical_threshold
            method = "total_variation_distance"

        return ColumnDriftResult(
            column_name=column_name,
            method=method,
            statistic=float(statistic),
            p_value=float(p_value),
            drift_detected=drift_detected,
            reference_size=int(reference_clean.shape[0]),
            current_size=int(current_clean.shape[0]),
        )

    @staticmethod
    def _categorical_distance(reference: pd.Series, current: pd.Series) -> float:
        reference_distribution = reference.astype(str).value_counts(normalize=True)
        current_distribution = current.astype(str).value_counts(normalize=True)
        categories = reference_distribution.index.union(current_distribution.index)

        reference_aligned = reference_distribution.reindex(categories, fill_value=0.0)
        current_aligned = current_distribution.reindex(categories, fill_value=0.0)
        return float(np.abs(reference_aligned - current_aligned).sum() / 2.0)
