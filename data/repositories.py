from pathlib import Path

import pandas as pd

from core.interfaces import DatasetRepository
from data.ingestion import CSVDataIngestion


class CSVDatasetRepository(DatasetRepository):
    def __init__(self, reference_path: str | Path, current_path: str | Path) -> None:
        self.ingestion = CSVDataIngestion(reference_path, current_path)

    def load_datasets(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return self.ingestion.load_datasets()

    def load_reference_dataset(self) -> pd.DataFrame:
        baseline_dataset, _ = self.load_datasets()
        return baseline_dataset

    def load_current_dataset(self) -> pd.DataFrame:
        _, incoming_dataset = self.load_datasets()
        return incoming_dataset
