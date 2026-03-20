from abc import ABC, abstractmethod

import pandas as pd

from core.entities import ColumnDriftResult


class DatasetRepository(ABC):
    @abstractmethod
    def load_reference_dataset(self) -> pd.DataFrame:
        """Return the baseline dataset used for drift comparison."""

    @abstractmethod
    def load_current_dataset(self) -> pd.DataFrame:
        """Return the current dataset under monitoring."""


class DriftDetector(ABC):
    @abstractmethod
    def analyze(
        self,
        column_name: str,
        reference: pd.Series,
        current: pd.Series,
        threshold: float,
    ) -> ColumnDriftResult:
        """Evaluate a single column for statistical drift."""

