class DriftGuardError(Exception):
    """Base exception for domain and application errors."""


class DataValidationError(DriftGuardError):
    """Raised when datasets are missing, invalid, or incompatible."""


class DatasetFileNotFoundError(DataValidationError):
    """Raised when an expected dataset file does not exist."""


class EmptyDatasetError(DataValidationError):
    """Raised when a dataset exists but contains no rows or columns."""


class SchemaMismatchError(DataValidationError):
    """Raised when datasets do not share an identical schema."""
