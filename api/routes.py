from fastapi import APIRouter, Depends, File, Form, UploadFile

from api.dependencies import build_detector_from_paths, build_detector_from_uploads
from api.schemas import (
    AnalyzeDriftRequest,
    DriftDetectionResponse,
    DriftMetricResponse,
    FeatureDriftMetricsResponse,
    FeatureDriftResponse,
)
from core.config import AppSettings, get_settings

router = APIRouter()


@router.get("/health")
def healthcheck(settings: AppSettings = Depends(get_settings)) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "environment": settings.environment,
        "debug": settings.debug,
    }


@router.post("/drift/analyze", response_model=DriftDetectionResponse)
def analyze_drift(
    payload: AnalyzeDriftRequest,
    settings: AppSettings = Depends(get_settings),
) -> DriftDetectionResponse:
    detector = build_detector_from_paths(
        settings=settings,
        reference_path=payload.reference_path,
        current_path=payload.current_path,
        use_predefined_paths=payload.use_predefined_paths,
    )
    report = detector.generate_report(dataset_name=payload.dataset_name)
    return _to_response(report)


@router.post("/drift/analyze/files", response_model=DriftDetectionResponse)
async def analyze_drift_from_files(
    reference_file: UploadFile = File(...),
    current_file: UploadFile = File(...),
    dataset_name: str | None = Form(None),
    settings: AppSettings = Depends(get_settings),
) -> DriftDetectionResponse:
    detector = await build_detector_from_uploads(
        settings=settings,
        reference_file=reference_file,
        current_file=current_file,
    )
    report = detector.generate_report(
        dataset_name=dataset_name or settings.runtime.uploaded_dataset_name,
    )
    return _to_response(report)


def _to_response(report) -> DriftDetectionResponse:
    return DriftDetectionResponse(
        dataset_name=report.dataset_name,
        generated_at=report.generated_at,
        total_features=report.total_features,
        drifted_features_count=len(report.drifted_features),
        stable_features_count=len(report.stable_features),
        features=[_to_feature_response(feature) for feature in report.features],
    )


def _to_feature_response(feature) -> FeatureDriftResponse:
    return FeatureDriftResponse(
        feature_name=feature.feature_name,
        feature_type=feature.feature_type,
        reference_size=feature.reference_size,
        current_size=feature.current_size,
        drift_detected=feature.drift_detected,
        supported=feature.supported,
        reason=feature.reason,
        metrics=FeatureDriftMetricsResponse(
            psi=_to_metric_response(feature.metrics.psi),
            ks=_to_metric_response(feature.metrics.ks),
            kl_divergence=_to_metric_response(feature.metrics.kl_divergence),
            chi_square=_to_metric_response(feature.metrics.chi_square),
            distribution_difference=_to_metric_response(feature.metrics.distribution_difference),
        ),
    )


def _to_metric_response(metric) -> DriftMetricResponse | None:
    if metric is None:
        return None

    return DriftMetricResponse(
        metric_name=metric.metric_name,
        value=metric.value,
        threshold=metric.threshold,
        drift_detected=metric.drift_detected,
        p_value=metric.p_value,
        interpretation=metric.interpretation,
    )
