from __future__ import annotations

import io
import logging

import numpy as np
import pandas as pd
import streamlit as st

from core.config import AppSettings, get_settings
from core.exceptions import DataValidationError
from data.ingestion import CSVDataIngestion
from drift.detectors import DriftDetector
from utils.dataset_validation import validate_compatible_datasets, validate_non_empty_dataset
from utils.logging import configure_logging, get_logger, log_event

logger = get_logger(__name__)


def main() -> None:
    settings = get_settings()
    configure_logging(settings.logging)

    st.set_page_config(
        page_title="DriftGuardAI Dashboard",
        layout="wide",
    )
    st.title("DriftGuardAI")
    st.caption("Visualize feature-level drift metrics across baseline and incoming datasets.")

    dataset_name, baseline_dataset, incoming_dataset = _render_sidebar(settings)
    if baseline_dataset is None or incoming_dataset is None:
        _render_empty_state(settings)
        return

    try:
        detector = DriftDetector(
            baseline_dataset=baseline_dataset,
            incoming_dataset=incoming_dataset,
            thresholds=settings.thresholds,
        )
        report = detector.generate_report(dataset_name=dataset_name)
    except DataValidationError as exc:
        st.error(str(exc))
        log_event(
            logger,
            logging.WARNING,
            "Dashboard drift detection failed.",
            event="dashboard_drift_detection_failed",
            dataset_name=dataset_name,
            error=str(exc),
        )
        return

    log_event(
        logger,
        logging.INFO,
        "Dashboard generated drift report.",
        event="dashboard_drift_report_generated",
        dataset_name=report.dataset_name,
        total_features=report.total_features,
        drifted_features=len(report.drifted_features),
    )

    metrics_frame = _build_metrics_frame(report)

    _render_summary(report)
    _render_metric_charts(metrics_frame)
    _render_drift_table(metrics_frame)
    _render_feature_distribution_section(
        baseline_dataset=baseline_dataset,
        incoming_dataset=incoming_dataset,
        metrics_frame=metrics_frame,
        settings=settings,
    )


def _render_sidebar(
    settings: AppSettings,
) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None]:
    st.sidebar.header("Data Source")
    dataset_mode = st.sidebar.radio(
        "Dataset Input",
        options=("Configured Paths", "Upload CSVs"),
    )
    default_dataset_name = (
        settings.runtime.default_dataset_name
        if dataset_mode == "Configured Paths"
        else settings.runtime.uploaded_dataset_name
    )
    dataset_name = st.sidebar.text_input("Dataset Name", value=default_dataset_name).strip() or default_dataset_name

    if dataset_mode == "Configured Paths":
        st.sidebar.caption(f"Baseline: `{settings.data.reference_dataset_path}`")
        st.sidebar.caption(f"Incoming: `{settings.data.current_dataset_path}`")
        if not st.sidebar.button("Run Drift Detection", use_container_width=True):
            return dataset_name, None, None
        return dataset_name, *_load_datasets_from_paths(settings)

    reference_file = st.sidebar.file_uploader("Baseline CSV", type=["csv"])
    current_file = st.sidebar.file_uploader("Incoming CSV", type=["csv"])
    if not st.sidebar.button("Run Drift Detection", use_container_width=True):
        return dataset_name, None, None
    if reference_file is None or current_file is None:
        st.sidebar.error("Upload both baseline and incoming CSV files.")
        return dataset_name, None, None
    return dataset_name, *_load_datasets_from_uploads(reference_file, current_file)


def _load_datasets_from_paths(settings: AppSettings) -> tuple[pd.DataFrame, pd.DataFrame]:
    ingestion = CSVDataIngestion(
        settings.data.reference_dataset_path,
        settings.data.current_dataset_path,
    )
    log_event(
        logger,
        logging.INFO,
        "Dashboard loading datasets from configured paths.",
        event="dashboard_load_datasets_from_paths",
        baseline_path=settings.data.reference_dataset_path,
        incoming_path=settings.data.current_dataset_path,
    )
    return ingestion.load_datasets()


def _load_datasets_from_uploads(
    reference_file,
    current_file,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_dataset = _read_uploaded_csv(reference_file, "Baseline")
    incoming_dataset = _read_uploaded_csv(current_file, "Incoming")
    validate_non_empty_dataset(baseline_dataset, "Baseline")
    validate_non_empty_dataset(incoming_dataset, "Incoming")
    validate_compatible_datasets(baseline_dataset, incoming_dataset)
    log_event(
        logger,
        logging.INFO,
        "Dashboard loaded datasets from uploaded files.",
        event="dashboard_load_datasets_from_uploads",
        baseline_filename=reference_file.name,
        incoming_filename=current_file.name,
        baseline_rows=int(baseline_dataset.shape[0]),
        incoming_rows=int(incoming_dataset.shape[0]),
    )
    return baseline_dataset, incoming_dataset


def _read_uploaded_csv(uploaded_file, dataset_name: str) -> pd.DataFrame:
    raw_bytes = uploaded_file.getvalue()
    if not raw_bytes:
        raise DataValidationError(f"{dataset_name} upload is empty.")

    try:
        return pd.read_csv(io.BytesIO(raw_bytes))
    except Exception as exc:  # pragma: no cover - defensive dashboard boundary
        raise DataValidationError(
            f"Failed to parse {dataset_name.lower()} upload '{uploaded_file.name}': {exc}",
        ) from exc


def _render_empty_state(settings: AppSettings) -> None:
    st.info(
        "Choose a data source in the sidebar, then run drift detection to inspect PSI, "
        "KS statistics, KL divergence, and feature distributions."
    )
    st.markdown(
        f"""
        Current configured dataset paths:
        - Baseline: `{settings.data.reference_dataset_path}`
        - Incoming: `{settings.data.current_dataset_path}`
        """
    )


def _render_summary(report) -> None:
    drift_ratio = (len(report.drifted_features) / report.total_features) if report.total_features else 0.0
    summary_columns = st.columns(4)
    summary_columns[0].metric("Total Features", report.total_features)
    summary_columns[1].metric("Drifted Features", len(report.drifted_features))
    summary_columns[2].metric("Stable Features", len(report.stable_features))
    summary_columns[3].metric("Drift Ratio", f"{drift_ratio:.1%}")

    if report.drifted_features:
        feature_names = ", ".join(feature.feature_name for feature in report.drifted_features[:8])
        suffix = "" if len(report.drifted_features) <= 8 else f" and {len(report.drifted_features) - 8} more"
        st.error(f"Significant drift detected in: {feature_names}{suffix}")
    else:
        st.success("No significant drift detected across the analyzed features.")


def _build_metrics_frame(report) -> pd.DataFrame:
    rows: list[dict] = []
    for feature in report.features:
        rows.append(
            {
                "feature_name": feature.feature_name,
                "feature_type": feature.feature_type,
                "drift_detected": feature.drift_detected,
                "supported": feature.supported,
                "reason": feature.reason,
                "psi": _metric_value(feature.metrics.psi),
                "psi_drift": _metric_flag(feature.metrics.psi),
                "ks_statistic": _metric_value(feature.metrics.ks),
                "ks_p_value": _metric_p_value(feature.metrics.ks),
                "ks_drift": _metric_flag(feature.metrics.ks),
                "kl_divergence": _metric_value(feature.metrics.kl_divergence),
                "kl_drift": _metric_flag(feature.metrics.kl_divergence),
                "chi_square_statistic": _metric_value(feature.metrics.chi_square),
                "chi_square_p_value": _metric_p_value(feature.metrics.chi_square),
                "chi_square_drift": _metric_flag(feature.metrics.chi_square),
                "distribution_difference": _metric_value(feature.metrics.distribution_difference),
                "distribution_difference_drift": _metric_flag(feature.metrics.distribution_difference),
            },
        )

    return pd.DataFrame(rows)


def _render_metric_charts(metrics_frame: pd.DataFrame) -> None:
    st.subheader("Metric Overview")
    numeric_metrics = metrics_frame[["feature_name", "psi", "ks_statistic", "kl_divergence"]].dropna(
        how="all",
        subset=["psi", "ks_statistic", "kl_divergence"],
    )
    if not numeric_metrics.empty:
        chart_columns = st.columns(3)
        chart_columns[0].caption("PSI by feature")
        chart_columns[0].bar_chart(numeric_metrics.set_index("feature_name")[["psi"]], use_container_width=True)
        chart_columns[1].caption("KS statistic by feature")
        chart_columns[1].bar_chart(
            numeric_metrics.set_index("feature_name")[["ks_statistic"]],
            use_container_width=True,
        )
        chart_columns[2].caption("KL divergence by feature")
        chart_columns[2].bar_chart(
            numeric_metrics.set_index("feature_name")[["kl_divergence"]],
            use_container_width=True,
        )

    categorical_metrics = metrics_frame[
        ["feature_name", "chi_square_statistic", "distribution_difference"]
    ].dropna(how="all", subset=["chi_square_statistic", "distribution_difference"])
    if not categorical_metrics.empty:
        chart_columns = st.columns(2)
        chart_columns[0].caption("Chi-square statistic by feature")
        chart_columns[0].bar_chart(
            categorical_metrics.set_index("feature_name")[["chi_square_statistic"]],
            use_container_width=True,
        )
        chart_columns[1].caption("Distribution difference by feature")
        chart_columns[1].bar_chart(
            categorical_metrics.set_index("feature_name")[["distribution_difference"]],
            use_container_width=True,
        )


def _render_drift_table(metrics_frame: pd.DataFrame) -> None:
    st.subheader("Feature Drift Table")
    styled_frame = metrics_frame.style.apply(_highlight_drifted_rows, axis=1)
    st.dataframe(styled_frame, use_container_width=True, hide_index=True)


def _render_feature_distribution_section(
    baseline_dataset: pd.DataFrame,
    incoming_dataset: pd.DataFrame,
    metrics_frame: pd.DataFrame,
    settings: AppSettings,
) -> None:
    st.subheader("Feature Distributions")
    feature_options = metrics_frame["feature_name"].tolist()
    if not feature_options:
        st.info("No features available for visualization.")
        return

    default_index = _default_feature_index(metrics_frame)
    selected_feature = st.selectbox(
        "Select a feature to inspect",
        options=feature_options,
        index=default_index,
    )

    baseline_series = baseline_dataset[selected_feature]
    incoming_series = incoming_dataset[selected_feature]
    feature_row = metrics_frame.loc[metrics_frame["feature_name"] == selected_feature].iloc[0]

    detail_columns = st.columns(2)
    detail_columns[0].markdown(
        f"**Feature Type:** `{feature_row['feature_type']}`  \n"
        f"**Drift Detected:** `{feature_row['drift_detected']}`"
    )
    detail_columns[1].markdown(
        f"**PSI:** `{_format_metric(feature_row['psi'])}`  \n"
        f"**KS Statistic:** `{_format_metric(feature_row['ks_statistic'])}`"
    )

    if pd.api.types.is_numeric_dtype(baseline_series) and not pd.api.types.is_bool_dtype(baseline_series):
        distribution_frame = _numeric_distribution_frame(
            baseline_series,
            incoming_series,
            bins=settings.thresholds.histogram_bins,
        )
        if distribution_frame.empty:
            st.warning("No numeric values are available for this feature distribution.")
            return
        st.caption("Numeric distribution comparison")
        st.bar_chart(
            distribution_frame.set_index("bucket")[["baseline_count", "incoming_count"]],
            use_container_width=True,
        )
    else:
        distribution_frame = _categorical_distribution_frame(baseline_series, incoming_series)
        st.caption("Categorical distribution comparison")
        st.bar_chart(
            distribution_frame.set_index("category")[["baseline_count", "incoming_count"]],
            use_container_width=True,
        )

    st.dataframe(distribution_frame, use_container_width=True, hide_index=True)


def _numeric_distribution_frame(
    baseline_series: pd.Series,
    incoming_series: pd.Series,
    bins: int,
) -> pd.DataFrame:
    baseline_values = pd.to_numeric(baseline_series, errors="coerce").dropna().to_numpy(dtype=float)
    incoming_values = pd.to_numeric(incoming_series, errors="coerce").dropna().to_numpy(dtype=float)
    if baseline_values.size == 0 or incoming_values.size == 0:
        return pd.DataFrame(columns=["bucket", "baseline_count", "incoming_count"])
    combined = np.concatenate((baseline_values, incoming_values))

    minimum = float(np.min(combined))
    maximum = float(np.max(combined))
    if minimum == maximum:
        minimum -= 0.5
        maximum += 0.5

    edges = np.linspace(minimum, maximum, bins + 1)
    baseline_counts, _ = np.histogram(baseline_values, bins=edges)
    incoming_counts, _ = np.histogram(incoming_values, bins=edges)
    labels = [
        f"{edges[index]:.2f} to {edges[index + 1]:.2f}"
        for index in range(len(edges) - 1)
    ]

    return pd.DataFrame(
        {
            "bucket": labels,
            "baseline_count": baseline_counts,
            "incoming_count": incoming_counts,
        },
    )


def _categorical_distribution_frame(
    baseline_series: pd.Series,
    incoming_series: pd.Series,
) -> pd.DataFrame:
    baseline_counts = baseline_series.fillna("__missing__").astype(str).value_counts()
    incoming_counts = incoming_series.fillna("__missing__").astype(str).value_counts()
    categories = baseline_counts.index.union(incoming_counts.index)
    baseline_aligned = baseline_counts.reindex(categories, fill_value=0)
    incoming_aligned = incoming_counts.reindex(categories, fill_value=0)
    return pd.DataFrame(
        {
            "category": categories,
            "baseline_count": baseline_aligned.to_numpy(),
            "incoming_count": incoming_aligned.to_numpy(),
        },
    )


def _metric_value(metric) -> float | None:
    return None if metric is None else metric.value


def _metric_p_value(metric) -> float | None:
    return None if metric is None else metric.p_value


def _metric_flag(metric) -> bool | None:
    return None if metric is None else metric.drift_detected


def _format_metric(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:.4f}"


def _highlight_drifted_rows(row: pd.Series) -> list[str]:
    if bool(row["drift_detected"]):
        return ["background-color: rgba(255, 99, 71, 0.18)"] * len(row)
    return [""] * len(row)


def _default_feature_index(metrics_frame: pd.DataFrame) -> int:
    drifted_indices = metrics_frame.index[metrics_frame["drift_detected"]].tolist()
    return drifted_indices[0] if drifted_indices else 0


if __name__ == "__main__":
    main()
