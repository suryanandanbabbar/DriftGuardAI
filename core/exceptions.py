class DriftGuardError(Exception):
    """Base exception for domain and application errors."""


class DataValidationError(DriftGuardError):
    """Raised when datasets are missing, invalid, or incompatible."""

