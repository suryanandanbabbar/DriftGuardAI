from fastapi import FastAPI

from api.routes import router
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
    return application


app = create_app()

