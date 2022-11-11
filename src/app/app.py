from fastapi import FastAPI

from logging import getLogger

from src.app import routers, routertest
from src.configurations import APIConfigurations

logger = getLogger(__name__)

app = FastAPI(
    title=APIConfigurations.title,
    description=APIConfigurations.description,
    version=APIConfigurations.version,
)

app.include_router(routers.router, prefix="", tags=[''])

# 서버 실행시
# uvicorn src.app.app:app --reload --host=0.0.0.0 --port=8001
# http://127.0.0.1:8001/docs#/