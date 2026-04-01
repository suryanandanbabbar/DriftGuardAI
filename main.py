from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router
from core.exceptions import DataValidationError, DriftGuardError
from core.config import get_settings
from utils.logging import configure_logging


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level)

    application = FastAPI(
        title=settings.api.title,
        version=settings.api.version,
        debug=settings.debug,
    )
    application.include_router(router, prefix=settings.api.prefix)

    @application.exception_handler(DataValidationError)
    async def handle_data_validation_error(
        request: Request,
        exc: DataValidationError,
    ) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @application.exception_handler(DriftGuardError)
    async def handle_drift_guard_error(
        request: Request,
        exc: DriftGuardError,
    ) -> JSONResponse:
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return application


app = create_app()
