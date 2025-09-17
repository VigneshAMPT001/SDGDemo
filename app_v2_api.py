from typing import Dict, Any, Optional, List
import io
import os
import zipfile
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import pandas as pd

# SDV imports
from sdv.io.local import CSVHandler
from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer

# CTGAN imports
from ctgan import CTGAN, TVAE


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


# ---------------------------
# In-memory stores (simple job/config registries)
# ---------------------------
SYNTHESIS_JOBS: Dict[str, Dict[str, Any]] = {}
QUALITY_JOBS: Dict[str, Dict[str, Any]] = {}
MODEL_CONFIGS: Dict[str, Any] = {}


# ---------------------------
# Helpers (reused logic from streamlit_app_v2)
# ---------------------------
AVAILABLE_DOMAINS: List[str] = ["Pharma", "BFSI", "Telecom"]


def load_domain_data(domain: str) -> Dict[str, pd.DataFrame]:
    connector = CSVHandler()
    if domain == "Pharma":
        data = connector.read(
            folder_name="data/pharma",
            file_names=["person.csv", "condition_era.csv"],
            read_csv_parameters={"parse_dates": False, "encoding": "latin-1"},
        )
        return {"person": data["person"], "condition_era": data["condition_era"]}
    return {}


def load_metadata(domain: str) -> Optional[Metadata]:
    if domain == "Pharma":
        return Metadata.load_from_json(filepath="metadata/metadata_pharma_v1.json")
    return None


def create_zip_from_tables(tables: Dict[str, pd.DataFrame]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in tables.items():
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            zip_file.writestr(f"{table_name}.csv", csv_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ---------------------------
# Pydantic models
# ---------------------------
class SDVParams(BaseModel):
    scale: float = Field(1.5, ge=0.1, le=5.0)


class SingleTableParams(BaseModel):
    epochs: int = Field(100, ge=10, le=2000)
    batch_size: int = Field(500, ge=16, le=4096)
    num_samples: int = Field(1000, ge=10, le=1_000_000)


class SynthesizeRequest(BaseModel):
    model_type: str = Field(
        ..., description="One of: 'SDV (Multi-table)', 'CTGAN', 'TVAE'"
    )
    domain: str = Field(..., description="Domain name, e.g., 'Pharma'")
    params: Dict[str, Any] = Field(default_factory=dict)


class ModelConfigRequest(BaseModel):
    model_type: str
    config: Dict[str, Any] = Field(default_factory=dict)


# ---------------------------
# Training utilities (no Streamlit dependency)
# ---------------------------


def train_sdv_model(
    data: Dict[str, pd.DataFrame], metadata: Metadata, params: Dict[str, Any]
):
    cleaned_data = drop_unknown_references(data, metadata)
    synthesizer = HMASynthesizer(metadata)
    synthesizer.fit(cleaned_data)
    scale = params.get("scale", 1.5)
    synthetic_data = synthesizer.sample(scale=scale)

    # Persist synthesizer artifact similarly to Streamlit app
    os.makedirs("synthesizer", exist_ok=True)
    synthesizer.save("synthesizer/synthesizer_pharma_v1.pkl")
    return synthetic_data, synthesizer


def train_ctgan_or_tvae(model_type: str, table: pd.DataFrame, params: Dict[str, Any]):
    epochs = params.get("epochs", 100)
    batch_size = params.get("batch_size", 500)
    num_samples = params.get("num_samples", max(100, len(table) * 2))

    if model_type == "CTGAN":
        model = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True)
    else:
        model = TVAE(epochs=epochs, batch_size=batch_size, verbose=True)

    # Heuristic for discrete columns
    discrete_columns: List[str] = params.get("discrete_columns") or [
        col
        for col in table.columns
        if table[col].dtype == "object" or table[col].nunique() < 20
    ]

    model.fit(table, discrete_columns=discrete_columns)
    synthetic_df = model.sample(num_samples)
    return synthetic_df, model


# ---------------------------
# Domain routes
# ---------------------------
@app.get("/domains")
def get_domains():
    return {"domains": AVAILABLE_DOMAINS, "implemented": ["Pharma"]}


@app.get("/domains/{domain}/datasets")
def get_domain_datasets(domain: str):
    if domain not in AVAILABLE_DOMAINS:
        raise HTTPException(status_code=404, detail="Unknown domain")

    data = load_domain_data(domain)
    if not data:
        raise HTTPException(
            status_code=404, detail="Datasets not available for this domain yet"
        )

    zip_bytes = create_zip_from_tables(data)
    file_name = f"{domain.lower()}_datasets.zip"
    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={file_name}"},
    )


@app.post("/domains/{domain}/synthesize")
def create_synthesis_job(domain: str, req: SynthesizeRequest):
    if req.domain != domain:
        raise HTTPException(status_code=400, detail="Request domain mismatch")

    if req.model_type not in ["SDV (Multi-table)", "CTGAN", "TVAE"]:
        raise HTTPException(status_code=400, detail="Unsupported model_type")

    data = load_domain_data(domain)
    if not data:
        raise HTTPException(status_code=404, detail="Domain data not available")

    job_id = str(uuid.uuid4())
    try:
        if req.model_type == "SDV (Multi-table)":
            metadata = load_metadata(domain)
            if metadata is None:
                raise HTTPException(status_code=400, detail="Metadata required for SDV")
            synthetic_data, _ = train_sdv_model(data, metadata, req.params)
            # multi-table dict[str, DataFrame]
            zip_bytes = create_zip_from_tables(synthetic_data)
            SYNTHESIS_JOBS[job_id] = {
                "status": "completed",
                "domain": domain,
                "model_type": req.model_type,
                "result_type": "zip",
                "bytes": zip_bytes,
                "filename": f"synthetic_{domain.lower()}_sdv.zip",
            }
        else:
            # single-table: pick first table
            first_table_name = list(data.keys())[0]
            table_df = data[first_table_name]
            synthetic_df, _ = train_ctgan_or_tvae(req.model_type, table_df, req.params)
            csv_buffer = io.StringIO()
            synthetic_df.to_csv(csv_buffer, index=False, encoding="utf-8")
            SYNTHESIS_JOBS[job_id] = {
                "status": "completed",
                "domain": domain,
                "model_type": req.model_type,
                "result_type": "csv",
                "bytes": csv_buffer.getvalue().encode("utf-8"),
                "filename": f"synthetic_{first_table_name}_{req.model_type.lower()}.csv",
            }
    except HTTPException:
        raise
    except Exception as e:
        SYNTHESIS_JOBS[job_id] = {"status": "failed", "error": str(e)}

    return {"job_id": job_id, "status": SYNTHESIS_JOBS[job_id]["status"]}


@app.get("/synthesis/{job_id}")
def get_synthesis_result(job_id: str):
    job = SYNTHESIS_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    if job["status"] != "completed":
        return JSONResponse(
            {"job_id": job_id, "status": job["status"], "error": job.get("error")}
        )

    if job["result_type"] == "zip":
        return StreamingResponse(
            io.BytesIO(job["bytes"]),
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename={job['filename']}"},
        )
    else:
        return StreamingResponse(
            io.BytesIO(job["bytes"]),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={job['filename']}"},
        )


# ---------------------------
# Data Quality placeholder routes (post->get)
# ---------------------------
@app.post("/quality-jobs")
def create_quality_job(payload: Dict[str, Any]):
    job_id = str(uuid.uuid4())
    QUALITY_JOBS[job_id] = {
        "status": "pending",
        "payload": payload,
        "message": "Quality assessment will be implemented later.",
    }
    return {"job_id": job_id, "status": "pending"}


@app.get("/quality-jobs/{job_id}")
def get_quality_job(job_id: str):
    job = QUALITY_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # For now, just echo placeholder
    return job


# ---------------------------
# Model routes
# ---------------------------
@app.get("/models")
def get_models():
    return {
        "models": [
            {"id": "sdv", "name": "SDV (Multi-table)", "capability": "multi-table"},
            {"id": "ctgan", "name": "CTGAN", "capability": "single-table"},
            {"id": "tvae", "name": "TVAE", "capability": "single-table"},
        ]
    }


@app.post("/models/config")
def post_model_config(cfg: ModelConfigRequest):
    MODEL_CONFIGS[cfg.model_type] = cfg.config
    return {"model_type": cfg.model_type, "saved": True}


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
