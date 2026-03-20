from fastapi import APIRouter, Depends, HTTPException

from api.dependencies import build_drift_use_case
from api.schemas import AnalyzeDriftRequest, ColumnDriftResponse, DriftAnalysisResponse
from core.config import AppSettings, get_settings
from core.exceptions import DataValidationError

router = APIRouter()


@router.get("/health")
def healthcheck(settings: AppSettings = Depends(get_settings)) -> dict[str, str | bool]:
    return {
        "status": "ok",
        "environment": settings.environment,
        "debug": settings.debug,
    }


@router.post("/drift/analyze", response_model=DriftAnalysisResponse)
def analyze_drift(
    payload: AnalyzeDriftRequest,
    settings: AppSettings = Depends(get_settings),
) -> DriftAnalysisResponse:
    use_case = build_drift_use_case(
        settings=settings,
        reference_path=payload.reference_path,
        current_path=payload.current_path,
    )

    try:
        report = use_case.execute(
            dataset_name=payload.dataset_name,
            columns=payload.columns or None,
        )
    except DataValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return DriftAnalysisResponse(
        dataset_name=report.dataset_name,
        generated_at=report.generated_at,
        total_columns=report.total_columns,
        drifted_columns=[_to_response(result) for result in report.drifted_columns],
        stable_columns=[_to_response(result) for result in report.stable_columns],
    )


def _to_response(result) -> ColumnDriftResponse:
    return ColumnDriftResponse(
        column_name=result.column_name,
        method=result.method,
        statistic=result.statistic,
        p_value=result.p_value,
        drift_detected=result.drift_detected,
        reference_size=result.reference_size,
        current_size=result.current_size,
    )

