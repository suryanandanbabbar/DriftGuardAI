from datetime import datetime, timezone
import logging

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
    categorical_distribution_difference,
    chi_square_test,
    kolmogorov_smirnov_test,
    kullback_leibler_divergence,
    population_stability_index,
)
from utils.dataset_validation import validate_compatible_datasets
from utils.logging import get_logger, log_event

logger = get_logger(__name__)


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
        log_event(
            logger,
            logging.INFO,
            "Initialized drift detector.",
            event="drift_detector_initialized",
            rows_baseline=int(self.baseline_dataset.shape[0]),
            rows_incoming=int(self.incoming_dataset.shape[0]),
            total_features=int(self.baseline_dataset.shape[1]),
        )

    def generate_report(self, dataset_name: str | None = None) -> FeatureDriftReport:
        features: list[FeatureDriftResult] = []
        resolved_dataset_name = dataset_name or get_settings().runtime.default_dataset_name
        log_event(
            logger,
            logging.INFO,
            "Starting drift detection report generation.",
            event="drift_detection_started",
            dataset_name=resolved_dataset_name,
            total_features=int(self.baseline_dataset.shape[1]),
        )

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

        report = FeatureDriftReport(
            dataset_name=resolved_dataset_name,
            generated_at=datetime.now(timezone.utc).isoformat(),
            features=features,
        )
        log_event(
            logger,
            logging.INFO,
            "Completed drift detection report generation.",
            event="drift_detection_completed",
            dataset_name=resolved_dataset_name,
            total_features=report.total_features,
            drifted_features=len(report.drifted_features),
            stable_features=len(report.stable_features),
        )
        return report

    def _analyze_feature(
        self,
        feature_name: str,
        baseline_feature: pd.Series,
        incoming_feature: pd.Series,
    ) -> FeatureDriftResult:
        if self._is_numeric_feature(baseline_feature) and self._is_numeric_feature(incoming_feature):
            return self._analyze_numeric_feature(feature_name, baseline_feature, incoming_feature)

        return self._analyze_categorical_feature(feature_name, baseline_feature, incoming_feature)

    def _analyze_numeric_feature(
        self,
        feature_name: str,
        baseline_feature: pd.Series,
        incoming_feature: pd.Series,
    ) -> FeatureDriftResult:
        feature_type = str(baseline_feature.dtype)

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
            log_event(
                logger,
                logging.WARNING,
                "Skipping numeric feature drift analysis due to invalid data.",
                event="numeric_feature_analysis_skipped",
                feature_name=feature_name,
                feature_type=feature_type,
                error=str(exc),
            )
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=feature_type,
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

        feature_result = FeatureDriftResult(
            feature_name=feature_name,
            feature_type=feature_type,
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
        self._log_feature_result(feature_result)
        return feature_result

    def _analyze_categorical_feature(
        self,
        feature_name: str,
        baseline_feature: pd.Series,
        incoming_feature: pd.Series,
    ) -> FeatureDriftResult:
        feature_type = str(baseline_feature.dtype)
        reference_size = int(baseline_feature.shape[0])
        current_size = int(incoming_feature.shape[0])

        try:
            chi_square_result = chi_square_test(
                baseline_feature,
                incoming_feature,
                significance_level=self.thresholds.categorical_chi_square_significance_level,
            )
            distance_value = categorical_distribution_difference(
                baseline_feature,
                incoming_feature,
            )
        except ValueError as exc:
            log_event(
                logger,
                logging.WARNING,
                "Skipping categorical feature drift analysis due to invalid data.",
                event="categorical_feature_analysis_skipped",
                feature_name=feature_name,
                feature_type=feature_type,
                error=str(exc),
            )
            return FeatureDriftResult(
                feature_name=feature_name,
                feature_type=feature_type,
                reference_size=reference_size,
                current_size=current_size,
                drift_detected=False,
                supported=False,
                reason=str(exc),
            )

        chi_square_metric = DriftMetricResult(
            metric_name="chi_square",
            value=float(chi_square_result.statistic),
            threshold=float(self.thresholds.categorical_chi_square_significance_level),
            drift_detected=chi_square_result.drift_detected,
            p_value=float(chi_square_result.p_value),
            interpretation=chi_square_result.interpretation,
        )
        distance_metric = DriftMetricResult(
            metric_name="distribution_difference",
            value=float(distance_value),
            threshold=float(self.thresholds.categorical_distance),
            drift_detected=bool(distance_value >= self.thresholds.categorical_distance),
            interpretation=(
                "Drift detected because categorical distribution difference exceeded "
                "the configured threshold."
                if distance_value >= self.thresholds.categorical_distance
                else "No categorical distribution-difference drift detected."
            ),
        )

        feature_result = FeatureDriftResult(
            feature_name=feature_name,
            feature_type=feature_type,
            reference_size=reference_size,
            current_size=current_size,
            drift_detected=any(
                metric.drift_detected for metric in (chi_square_metric, distance_metric)
            ),
            metrics=FeatureDriftMetrics(
                chi_square=chi_square_metric,
                distribution_difference=distance_metric,
            ),
        )
        self._log_feature_result(feature_result)
        return feature_result

    @staticmethod
    def _is_numeric_feature(feature: pd.Series) -> bool:
        return pd.api.types.is_numeric_dtype(feature) and not pd.api.types.is_bool_dtype(feature)

    @staticmethod
    def _log_feature_result(feature_result: FeatureDriftResult) -> None:
        metric_payload = {
            metric.metric_name: {
                "value": metric.value,
                "threshold": metric.threshold,
                "drift_detected": metric.drift_detected,
                "p_value": metric.p_value,
            }
            for metric in (
                feature_result.metrics.psi,
                feature_result.metrics.ks,
                feature_result.metrics.kl_divergence,
                feature_result.metrics.chi_square,
                feature_result.metrics.distribution_difference,
            )
            if metric is not None
        }
        level = logging.WARNING if feature_result.drift_detected else logging.INFO
        log_event(
            logger,
            level,
            "Computed drift analysis for feature.",
            event="feature_drift_analyzed",
            feature_name=feature_result.feature_name,
            feature_type=feature_result.feature_type,
            drift_detected=feature_result.drift_detected,
            supported=feature_result.supported,
            reference_size=feature_result.reference_size,
            current_size=feature_result.current_size,
            metrics=metric_payload,
            reason=feature_result.reason,
        )


class StatisticalDriftDetector(BaseDriftDetector):
    def __init__(self, categorical_threshold: float | None = None) -> None:
        self.categorical_threshold = (
            categorical_threshold
            if categorical_threshold is not None
            else get_settings().thresholds.categorical_distance
        )
        log_event(
            logger,
            logging.DEBUG,
            "Initialized statistical drift detector.",
            event="statistical_drift_detector_initialized",
            categorical_threshold=float(self.categorical_threshold),
        )

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
        return categorical_distribution_difference(reference, current)
