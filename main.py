# main.py
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import io
import os
import zipfile
import tempfile
import pandas as pd
import json

# SDV imports (same as your streamlit)
from sdv.io.local import CSVHandler
from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer

# CTGAN imports
from ctgan import CTGAN, TVAE

app = FastAPI(title="DataSyn API", version="1.0")

# --- Helpers ---

def create_zip_bytes(synthetic_data: Dict[str, pd.DataFrame]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for k, df in synthetic_data.items():
            csv_buf = io.StringIO()
            df.to_csv(csv_buf, index=False, encoding="utf-8")
            zf.writestr(f"{k}_synthetic.csv", csv_buf.getvalue())
    buf.seek(0)
    return buf.getvalue()

def load_domain_data_pharma() -> Dict[str, pd.DataFrame]:
    # Assumes repository contains data/pharma/person.csv and condition_era.csv
    base = PathLike = os.path
    path = os.path.join(os.getcwd(), "data", "pharma")
    handler = CSVHandler()
    data = handler.read(
        folder_name=path,
        file_names=["person.csv", "condition_era.csv"],
        read_csv_parameters={"parse_dates": False, "encoding": "latin-1"},
    )
    return {"person": data["person"], "condition_era": data["condition_era"]}

def load_metadata_pharma() -> Optional[Metadata]:
    try:
        return Metadata.load_from_json(filepath="metadata/metadata_pharma_v1.json")
    except Exception:
        return None

# --- Pydantic models for requests ---

class SDVParams(BaseModel):
    scale: float = 1.5

class GANParams(BaseModel):
    epochs: int = 100
    batch_size: int = 500
    num_samples: int = 1000

# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/domain/load")
def domain_load(domain: str = "Pharma"):
    """Load pre-configured domain data and metadata (if available)."""
    if domain.lower() == "pharma":
        try:
            data = load_domain_data_pharma()
            metadata = load_metadata_pharma()
            tables_info = {k: {"rows": len(df), "cols": len(df.columns)} for k, df in data.items()}
            return {"domain": "Pharma", "tables": tables_info, "has_metadata": metadata is not None}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=404, detail="Domain not configured")

@app.post("/upload_csvs")
async def upload_csvs(files: List[UploadFile] = File(...)):
    """
    Upload one or more CSVs. Returns a JSON listing tables and shapes.
    The files are kept in-memory; you can pass them to the /generate endpoint.
    """
    stored = {}
    for f in files:
        if not f.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only CSV files are accepted")
        df = pd.read_csv(io.BytesIO(await f.read()))
        stored[os.path.splitext(f.filename)[0]] = df
    # store briefly into a temp directory and return a lightweight handle (we'll return data directly)
    # For simplicity, return shapes and base64 not included
    shapes = {k: {"rows": v.shape[0], "cols": v.shape[1]} for k, v in stored.items()}
    # Return first 5 rows preview as JSON (to avoid huge payloads)
    previews = {k: v.head(5).to_dict(orient="records") for k, v in stored.items()}
    # Keep uploaded data temporarily in /tmp with a generated id optional - here we return full previews only.
    return {"uploaded_tables": list(stored.keys()), "shapes": shapes, "previews": previews}

@app.post("/generate/sdv")
def generate_sdv(domain: str = Form(...), scale: float = Form(1.5)):
    """
    Generate SDV multi-table synthetic data using domain metadata.
    domain must be 'Pharma' for now.
    Returns a ZIP containing CSVs.
    """
    if domain.lower() != "pharma":
        raise HTTPException(status_code=400, detail="Only 'Pharma' domain supported for SDV currently")
    # load domain data & metadata
    try:
        data = load_domain_data_pharma()
        metadata = load_metadata_pharma()
        if metadata is None:
            raise HTTPException(status_code=400, detail="Metadata for domain not found")
        # Clean data
        cleaned = drop_unknown_references(data, metadata)
        # Init synthesizer
        synthesizer = HMASynthesizer(metadata)
        synthesizer.fit(cleaned)
        synthetic = synthesizer.sample(scale=scale)
        zip_bytes = create_zip_bytes(synthetic)
        return StreamingResponse(io.BytesIO(zip_bytes), media_type="application/zip",
                                 headers={"Content-Disposition": f"attachment; filename=synthetic_{domain}_sdv.zip"})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/gan")
async def generate_gan(
    model_type: str = Form(...),  # 'CTGAN' or 'TVAE'
    file: UploadFile = File(...),
    epochs: int = Form(100),
    batch_size: int = Form(500),
    num_samples: int = Form(1000),
):
    """
    Train a CTGAN/TVAE on a single uploaded CSV and return generated CSV.
    """
    if model_type not in ("CTGAN", "TVAE"):
        raise HTTPException(status_code=400, detail="model_type must be CTGAN or TVAE")
    try:
        file_bytes = await file.read()
        df = pd.read_csv(io.BytesIO(file_bytes))
        # heuristics to detect discrete columns
        potential_discrete = [c for c in df.columns if df[c].dtype == "object" or df[c].nunique() < 20]
        # initialize model
        if model_type == "CTGAN":
            model = CTGAN(epochs=epochs, batch_size=batch_size, verbose=False)
        else:
            model = TVAE(epochs=epochs, batch_size=batch_size, verbose=False)
        model.fit(df, discrete_columns=potential_discrete)
        synthetic = model.sample(num_samples)
        csv_buf = io.StringIO()
        synthetic.to_csv(csv_buf, index=False, encoding="utf-8")
        return StreamingResponse(io.StringIO(csv_buf.getvalue()), media_type="text/csv",
                                 headers={"Content-Disposition": f"attachment; filename=synthetic_{model_type}.csv"})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
