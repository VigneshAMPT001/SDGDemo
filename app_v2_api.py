from typing import Dict, Any, Optional, List
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

# Import shared models and data structures
from api_routers.shared import (
    QUALITY_JOBS,
    MODEL_CONFIGS,
    AVAILABLE_DOMAINS,
    SDVParams,
    SingleTableParams,
    SynthesizeRequest,
    ModelConfigRequest,
    DatasetTable,
    DatasetResponse,
    QualityJobRequest,
    VisualizationQuery,
)

from api_routers.generate_synthetic_data import router as synthetic_data_router
from api_routers.models import router as models_router
from api_routers.run_quality_checks import router as quality_jobs_router
from api_routers.domains import router as domain_router
from api_routers.validate_dataset import router as validation_router

ROUTERS = [
    synthetic_data_router,
    models_router,
    quality_jobs_router,
    domain_router,
    validation_router,
]

# ---------------------------
# App setup
# ---------------------------
app = FastAPI(title="DataSyn API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(synthetic_data_router)
app.include_router(models_router)
app.include_router(quality_jobs_router)
app.include_router(domain_router)
app.include_router(validation_router)


# ---------------------------
# Root
# ---------------------------
@app.get("/")
def read_root():
    return {"name": "DataSyn API", "version": "2.0"}


# ---------------------------
# Uvicorn entrypoint (optional)
# ---------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app_v2_api:app", host="0.0.0.0", port=8000, reload=True)
