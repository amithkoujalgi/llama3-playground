import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi

from llama3_playground.server.routers.inference import router as inference_router
from llama3_playground.server.routers.models import router as models_router
from llama3_playground.server.routers.ocr_extraction import router as ocr_extraction_router
from llama3_playground.server.routers.training import router as training_router

app_title = 'Llama3 Playground'
app_version = '0.0.1'
app_desc = (
    f"<p>"
    f"REST API server for model training and serving."
    f"</p>"
)

origins = os.getenv("CORS_ORIGINS", "*").split(" ")  # provide all the allowed origins as space separated
app = FastAPI(
    title=app_title,
    version=app_version,
    description=app_desc,
    swagger_ui_parameters={
        # "defaultModelsExpandDepth": -1,  # To hide the schema section from swagger docs
        "requestSnippetsEnabled": True,
        "displayRequestDuration": True,
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "syntaxHighlight": True,
        "syntaxHighlight.activate": True,
        "syntaxHighlight.theme": "tomorrow-night",
        "tryItOutEnabled": True,
    },
)


def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title=app_title,
        version=app_version,
        description=app_desc,
        routes=app.routes,
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi

app.include_router(
    training_router,
    prefix="/api/train",
    tags=["train"],
)

app.include_router(
    inference_router,
    prefix="/api/infer",
    tags=["inference"],
)

app.include_router(
    ocr_extraction_router,
    prefix="/api/ocr",
    tags=["ocr"],
)

app.include_router(
    models_router,
    prefix="/api/models",
    tags=["models"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
