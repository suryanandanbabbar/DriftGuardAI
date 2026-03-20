import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from core.entities import ColumnDriftResult
from core.interfaces import DriftDetector


class StatisticalDriftDetector(DriftDetector):
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
            statistic, p_value = ks_2samp(reference_clean.to_numpy(), current_clean.to_numpy())
            drift_detected = p_value < threshold
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

