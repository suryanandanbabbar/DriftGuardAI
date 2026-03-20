from __future__ import annotations

import pandas as pd

from core.exceptions import EmptyDatasetError, SchemaMismatchError


def validate_non_empty_dataset(dataset: pd.DataFrame, dataset_name: str) -> None:
    if dataset.empty or dataset.columns.empty:
        raise EmptyDatasetError(
            f"{dataset_name} dataset is empty. Both rows and columns are required for ingestion.",
        )


def ensure_identical_columns(
    baseline_dataset: pd.DataFrame,
    incoming_dataset: pd.DataFrame,
) -> None:
    baseline_columns = list(baseline_dataset.columns)
    incoming_columns = list(incoming_dataset.columns)

    if baseline_columns != incoming_columns:
        missing_in_incoming = [column for column in baseline_columns if column not in incoming_columns]
        extra_in_incoming = [column for column in incoming_columns if column not in baseline_columns]
        raise SchemaMismatchError(
            "Dataset columns do not match. "
            f"Missing in incoming: {missing_in_incoming or 'none'}. "
            f"Unexpected in incoming: {extra_in_incoming or 'none'}. "
            f"Baseline order: {baseline_columns}. Incoming order: {incoming_columns}.",
        )


def ensure_identical_dtypes(
    baseline_dataset: pd.DataFrame,
    incoming_dataset: pd.DataFrame,
) -> None:
    dtype_mismatches: list[str] = []

    for column_name in baseline_dataset.columns:
        baseline_dtype = baseline_dataset[column_name].dtype
        incoming_dtype = incoming_dataset[column_name].dtype
        if baseline_dtype != incoming_dtype:
            dtype_mismatches.append(
                f"{column_name}: baseline={baseline_dtype}, incoming={incoming_dtype}",
            )

    if dtype_mismatches:
        raise SchemaMismatchError(
            "Dataset column data types do not match: " + "; ".join(dtype_mismatches),
        )


def validate_compatible_datasets(
    baseline_dataset: pd.DataFrame,
    incoming_dataset: pd.DataFrame,
) -> None:
    validate_non_empty_dataset(baseline_dataset, "Baseline")
    validate_non_empty_dataset(incoming_dataset, "Incoming")
    ensure_identical_columns(baseline_dataset, incoming_dataset)
    ensure_identical_dtypes(baseline_dataset, incoming_dataset)
