from fastapi import FastAPI

from logging import getLogger

from src.app import routers
from src.configurations import APIConfigurations

logger = getLogger(__name__)

app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

app.include_router(routers.router, prefix="", tags=[''])

# uvicorn src.app.app:app --reload --host=0.0.0.0 --port=8001
