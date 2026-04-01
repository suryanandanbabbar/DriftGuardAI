import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from api.routes import router
from core.exceptions import DataValidationError, DriftGuardError
from core.config import get_settings
from utils.logging import configure_logging, get_logger, log_event

logger = get_logger(__name__)


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.logging)
    log_event(
        logger,
        logging.INFO,
        "Creating FastAPI application.",
        event="application_startup",
        environment=settings.environment,
        debug=settings.debug,
        api_prefix=settings.api.prefix,
        log_level=settings.logging.level,
        structured_logging=settings.logging.structured,
    )

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
        log_event(
            logger,
            logging.WARNING,
            "Handled data validation error.",
            event="data_validation_error",
            path=str(request.url.path),
            error=str(exc),
        )
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @application.exception_handler(DriftGuardError)
    async def handle_drift_guard_error(
        request: Request,
        exc: DriftGuardError,
    ) -> JSONResponse:
        log_event(
            logger,
            logging.ERROR,
            "Handled application error.",
            event="application_error",
            path=str(request.url.path),
            error=str(exc),
        )
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    return application


app = create_app()
