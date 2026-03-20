from pathlib import Path

import pandas as pd

from core.exceptions import DataValidationError
from core.interfaces import DatasetRepository


class CSVDatasetRepository(DatasetRepository):
    def __init__(self, reference_path: str | Path, current_path: str | Path) -> None:
        self.reference_path = self._resolve_path(reference_path)
        self.current_path = self._resolve_path(current_path)

    def load_reference_dataset(self) -> pd.DataFrame:
        return self._load_csv(self.reference_path)

    def load_current_dataset(self) -> pd.DataFrame:
        return self._load_csv(self.current_path)

    def _load_csv(self, path: Path) -> pd.DataFrame:
        if not path.exists():
            raise DataValidationError(f"Dataset not found: {path}")

        try:
            return pd.read_csv(path)
        except Exception as exc:  # pragma: no cover - defensive adapter boundary
            raise DataValidationError(f"Failed to load dataset from {path}: {exc}") from exc

    @staticmethod
    def _resolve_path(path_value: str | Path) -> Path:
        path = Path(path_value)
        if path.is_absolute():
            return path
        return Path.cwd() / path

