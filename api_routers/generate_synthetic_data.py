from fastapi import APIRouter, HTTPException
from fastapi import File, Form, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from datetime import datetime
from typing import List, Dict, Any, Optional
import json, io, uuid, os, tempfile
from sdv.metadata import Metadata
import pandas as pd

from api_routers.shared import (
    SynthesizeRequest,
)
from api_routers.route_utils import (
    load_domain_data,
    create_zip_from_tables,
    train_sdv_model,
    train_ctgan_or_tvae,
    load_domain_metadata,
)
from dbstore.connect import (
    create_connection,
    insert_job_record,
    get_job_by_id,
    fetch_query,
)

router = APIRouter()

router = APIRouter(prefix="/synthesize", tags=["synthesize"])


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def validate_synthesis_request(domain: str, req: SynthesizeRequest) -> None:
    """Validate synthesis request parameters."""
    if req.domain != domain:
        raise HTTPException(status_code=400, detail="Request domain mismatch")

    if req.model_type not in ["SDV (Multi-table)", "CTGAN", "TVAE"]:
        raise HTTPException(status_code=400, detail="Unsupported model_type")


def _persist_synthesis_job(
    *,
    file_bytes: Optional[bytes],
    file_name: Optional[str],
    file_type: Optional[str],
    domain: Optional[str],
    model_used: Optional[str],
    num_files: Optional[int],
    status: str = "completed",
    version: int = 1,
) -> str:
    """Persist a synthesis job into DB and return the job_id.

    Raises HTTPException on failure.
    """
    job_id = str(uuid.uuid4())
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        insert_ok = insert_job_record(
            conn,
            job_id=job_id,
            file_name=file_name,
            file_bytes=file_bytes,
            num_files=num_files,
            status=status,
            model_used=model_used,
            version=version,
            file_size=len(file_bytes) if file_bytes is not None else None,
            file_type=file_type,
            domain=domain,
            use_case=None,
        )
        if not insert_ok:
            raise HTTPException(status_code=500, detail="Failed to persist job")
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return job_id


def process_sdv_synthesis(
    domain: str, usecase: str, data: Dict[str, pd.DataFrame], req: SynthesizeRequest
) -> Dict[str, Any]:
    """Process SDV multi-table synthesis."""
    metadata = load_domain_metadata(domain, usecase)
    if metadata is None:
        raise HTTPException(status_code=400, detail="Metadata required for SDV")

    synthetic_data, _ = train_sdv_model(data, metadata, req.params)
    zip_bytes = create_zip_from_tables(synthetic_data)

    return {
        "status": "completed",
        "domain": domain,
        "model_type": req.model_type,
        "result_type": "zip",
        "bytes": zip_bytes,
        "filename": f"synthetic_{domain.lower()}_sdv.zip",
        "file_type": "application/zip",
    }


def process_single_table_synthesis(
    data: Dict[str, pd.DataFrame], req: SynthesizeRequest
) -> Dict[str, Any]:
    """Process single-table synthesis (CTGAN/TVAE)."""
    first_table_name = list(data.keys())[0]
    table_df = data[first_table_name]
    synthetic_df, _ = train_ctgan_or_tvae(req.model_type, table_df, req.params)

    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False, encoding="utf-8")

    return {
        "status": "completed",
        "domain": req.domain,
        "model_type": req.model_type,
        "result_type": "csv",
        "bytes": csv_buffer.getvalue().encode("utf-8"),
        "filename": f"synthetic_{first_table_name}_{req.model_type.lower()}.csv",
        "file_type": "text/csv",
    }


def create_synthesis_job_result(
    domain: str, usecase: str, req: SynthesizeRequest
) -> Dict[str, Any]:
    """Create synthesis job, persist to DB, and return result stub."""
    data = load_domain_data(domain, usecase)
    if not data:
        raise HTTPException(status_code=404, detail="Domain data not available")

    try:
        if req.model_type == "SDV (Multi-table)":
            result = process_sdv_synthesis(domain, usecase, data, req)
        else:
            result = process_single_table_synthesis(data, req)

        # Persist to DB (centralized)
        job_id = _persist_synthesis_job(
            file_bytes=result["bytes"],
            file_name=result["filename"],
            file_type=result.get("file_type"),
            domain=result["domain"],
            model_used=req.model_type,
            num_files=len(data),
            status=result["status"],
        )

        return {"job_id": job_id, "status": result["status"]}

    except HTTPException:
        raise
    except Exception as e:
        # Persist failed job with error status (no bytes)
        try:
            _persist_synthesis_job(
                file_bytes=None,
                file_name=None,
                file_type=None,
                domain=domain,
                model_used=req.model_type,
                num_files=len(data) if data else None,
                status="failed",
            )
        except Exception:
            pass
        return {"status": "failed", "error": str(e)}


def parse_model_params(params: Optional[str]) -> Dict[str, Any]:
    """Parse model parameters from JSON string."""
    model_params: Dict[str, Any] = {}
    if params:
        try:
            model_params = json.loads(params)
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid params JSON")
    return model_params


async def read_uploaded_files(files: List[UploadFile]) -> Dict[str, pd.DataFrame]:
    """Read and parse uploaded CSV files."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    tables: Dict[str, pd.DataFrame] = {}
    for uf in files:
        try:
            content = await uf.read()
            df = pd.read_csv(io.BytesIO(content))
            table_name = os.path.splitext(os.path.basename(uf.filename or "table"))[0]
            tables[table_name] = df
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to read {uf.filename}: {str(e)}"
            )
    return tables


def infer_discrete_columns(
    df: pd.DataFrame, model_params: Dict[str, Any]
) -> Dict[str, Any]:
    """Infer discrete columns for single-table models."""
    if "discrete_columns" not in model_params:
        model_params["discrete_columns"] = [
            c for c in df.columns if df[c].dtype == "object" or df[c].nunique() < 20
        ]
    return model_params


async def load_metadata_from_upload(metadata_json: UploadFile) -> Metadata:
    """Load metadata from uploaded JSON file."""
    try:
        meta_bytes = await metadata_json.read()
        with tempfile.NamedTemporaryFile("wb", delete=False, suffix=".json") as tmp:
            tmp.write(meta_bytes)
            tmp_path = tmp.name
        try:
            metadata = Metadata.load_from_json(filepath=tmp_path)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
        return metadata
    except Exception as e:
        raise HTTPException(
            status_code=400, detail=f"Failed to read metadata_json: {str(e)}"
        )


def process_single_table_custom(
    tables: Dict[str, pd.DataFrame], model_type: str, model_params: Dict[str, Any]
) -> StreamingResponse:
    """Process single-table custom synthesis."""
    chosen_model = model_type or "CTGAN"
    if chosen_model not in ["CTGAN", "TVAE"]:
        raise HTTPException(
            status_code=400,
            detail="model_type must be CTGAN or TVAE for single-table",
        )

    table_name = list(tables.keys())[0]
    df = tables[table_name]

    model_params = infer_discrete_columns(df, model_params)
    synthetic_df, _ = train_ctgan_or_tvae(chosen_model, df, model_params)

    csv_buffer = io.StringIO()
    synthetic_df.to_csv(csv_buffer, index=False, encoding="utf-8")
    file_bytes = csv_buffer.getvalue().encode("utf-8")
    file_name = f"synthetic_{table_name}_{chosen_model.lower()}.csv"
    file_type = "text/csv"

    # Persist job for custom single-table (domain marked as 'Custom')
    job_id = _persist_synthesis_job(
        file_bytes=file_bytes,
        file_name=file_name,
        file_type=file_type,
        domain="Custom",
        model_used=chosen_model,
        num_files=len(tables),
        status="completed",
    )

    # Include job id in headers for traceability while streaming the file
    return StreamingResponse(
        io.BytesIO(file_bytes),
        media_type=file_type,
        headers={
            "Content-Disposition": f"attachment; filename={os.path.splitext(file_name)[0]}{datetime.now().strftime('%Y%m%d%H%M%S')}.csv",
            "X-Job-Id": job_id,
        },
    )


async def process_multi_table_custom(
    tables: Dict[str, pd.DataFrame],
    metadata_json: Optional[UploadFile],
    model_params: Dict[str, Any],
) -> StreamingResponse:
    """Process multi-table custom synthesis."""
    if metadata_json is not None:
        # Use provided metadata
        metadata = await load_metadata_from_upload(metadata_json)
    else:
        # Auto-detect metadata from tables
        metadata = Metadata.detect_from_dataframes(data=tables)

    synthetic_data, _ = train_sdv_model(tables, metadata, model_params)
    zip_bytes = create_zip_from_tables(synthetic_data)
    file_name = "synthetic_multi_table_sdv.zip"
    file_type = "application/zip"

    # Persist job for custom multi-table (domain marked as 'Custom')
    job_id = _persist_synthesis_job(
        file_bytes=zip_bytes,
        file_name=file_name,
        file_type=file_type,
        domain="Custom",
        model_used="SDV (Multi-table)",
        num_files=len(tables),
        status="completed",
    )

    return StreamingResponse(
        io.BytesIO(zip_bytes),
        media_type=file_type,
        headers={
            "Content-Disposition": f"attachment; filename=synthetic_multi_table_sdv{datetime.now().strftime('%Y%m%d%H%M%S')}.zip",
            "X-Job-Id": job_id,
        },
    )


def get_all_jobs_result(domain: str):
    """Get all jobs for the specified domain."""
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        query = (
            "SELECT job_id, created_at, file_name, file_size, "
            "num_files, status, model_used, version, file_type, domain, use_case "
            "FROM synthetic_jobs WHERE domain = %s"
        )
        jobs = fetch_query(conn, query, (domain,))
    finally:
        try:
            conn.close()
        except Exception:
            pass

    return jobs


def download_synthetic_dataset_result(domain: str, job_id: str):
    """Download a synthetic dataset by job_id and domain."""
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")

    try:
        query = "SELECT csv_bytes, file_type FROM synthetic_jobs WHERE job_id = %s AND domain = %s"
        rows = fetch_query(conn, query, (job_id, domain))
        if not rows:
            raise HTTPException(status_code=404, detail="Job not found")

        # if rows is dict-based
        if isinstance(rows[0], dict):
            csv_bytes = rows[0]["csv_bytes"]
            file_type = rows[0]["file_type"]
        else:
            csv_bytes = rows[0][0]
            file_type = rows[0][1]
        return StreamingResponse(io.BytesIO(csv_bytes), media_type=file_type)
    finally:
        try:
            conn.close()
        except Exception:
            pass


# =============================================================================
# ROUTE FUNCTIONS
# =============================================================================


@router.post("/{domain}/generate_synthetic_dataset")
def create_synthesis_job(domain: str, req: SynthesizeRequest):
    """Create a synthesis job for the specified domain."""
    validate_synthesis_request(domain, req)
    return create_synthesis_job_result(domain, req)


@router.get("/{domain}/download_synthetic_dataset/{job_id}")
def download_synthetic_dataset(domain: str, job_id: str):
    """Download a synthetic dataset by job_id and domain."""
    return download_synthetic_dataset_result(domain, job_id)


@router.get("/{domain}/get_all_jobs")
def get_all_jobs(domain: str):
    """Get all jobs for the specified domain."""
    return get_all_jobs_result(domain)


@router.get("/{job_id}")
def get_synthesis_result(job_id: str):
    """Get the result of a synthesis job from DB."""
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        job = get_job_by_id(conn, job_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job.get("status")
    if status != "completed":
        return JSONResponse({"job_id": job_id, "status": status})

    file_name = job.get("file_name") or "synthetic_output"
    csv_bytes = job.get("csv_bytes") or b""
    file_type = job.get("file_type")

    # Determine media_type: prefer stored file_type; fallback by extension
    media_type = file_type
    if not media_type:
        is_zip = str(file_name).lower().endswith(".zip")
        media_type = "application/zip" if is_zip else "text/csv"

    ts_suffix = datetime.now().strftime("%Y%m%d%H%M%S")
    disposition_name = f"{file_name}{ts_suffix}"

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={disposition_name}"},
    )


@router.post("/custom")
async def synthesize_custom(
    files: List[UploadFile] = File(..., description="One or more CSV files"),
    metadata_json: Optional[UploadFile] = File(
        default=None,
        description="Optional metadata JSON for multi-table SDV (auto-detected if not provided)",
    ),
    model_type: Optional[str] = Form(
        default=None,
        description="Optional: 'CTGAN' or 'TVAE' for single-table; SDV auto for multi-table",
    ),
    params: Optional[str] = Form(
        default=None, description="Optional JSON string of model parameters"
    ),
):
    """Synthesize custom datasets (single or multi-table)."""
    # Parse parameters
    model_params = parse_model_params(params)

    # Read uploaded files
    tables = await read_uploaded_files(files)

    # Handle single-table case
    if len(tables) == 1:
        return process_single_table_custom(tables, model_type, model_params)

    # Handle multi-table case - metadata is now optional
    return await process_multi_table_custom(tables, metadata_json, model_params)
