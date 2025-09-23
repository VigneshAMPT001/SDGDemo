from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
import json
import uuid
from sdv.evaluation.multi_table import evaluate_quality
from typing import Optional, Dict, Any
from api_routers.shared import (
    QUALITY_JOBS,
    SYNTHESIS_JOBS,
    QualityJobRequest,
)

from api_routers.route_utils import (
    load_domain_data,
    read_multi_table_zip_bytes,
    load_domain_metadata,
)

router = APIRouter()

router = APIRouter(prefix="/quality_runs", tags=["quality_runs"])


# ---------------------------
# Data Quality routes (evaluate + visualize)
# ---------------------------
@router.post("/")
def create_quality_job(payload: QualityJobRequest):
    domain = payload.domain
    synth_job_id = payload.synthesis_job_id

    synth_job = SYNTHESIS_JOBS.get(synth_job_id)
    if not synth_job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")
    if synth_job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Synthesis job not completed")
    if synth_job.get("result_type") != "zip":
        raise HTTPException(
            status_code=400,
            detail="Quality evaluation currently supports SDV multi-table (zip) results only",
        )

    data = load_domain_data(domain)
    if not data:
        raise HTTPException(status_code=404, detail="Domain data not available")
    metadata = load_domain_metadata(domain)
    if metadata is None:
        raise HTTPException(
            status_code=400, detail="Metadata required for quality evaluation"
        )

    job_id = str(uuid.uuid4())
    try:
        synthetic_data = read_multi_table_zip_bytes(synth_job["bytes"])
        report = evaluate_quality(
            real_data=data, synthetic_data=synthetic_data, metadata=metadata
        )
        score: Optional[float] = None
        try:
            score = report.get_score()
        except Exception:
            score = None
        QUALITY_JOBS[job_id] = {
            "status": "completed",
            "domain": domain,
            "synthesis_job_id": synth_job_id,
            "score": score,
            "report": report,  # keep in-memory for visualization
            "message": "Quality evaluation completed",
        }
    except Exception as e:
        QUALITY_JOBS[job_id] = {
            "status": "failed",
            "domain": domain,
            "synthesis_job_id": synth_job_id,
            "error": str(e),
        }

    return {
        "job_id": job_id,
        "status": QUALITY_JOBS[job_id]["status"],
        "score": QUALITY_JOBS[job_id].get("score"),
    }


@router.get("/{job_id}")
def get_quality_job(job_id: str):
    job = QUALITY_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    # Serialize minimal info; do not attempt to serialize the report object
    response: Dict[str, Any] = {
        "job_id": job_id,
        "status": job.get("status"),
        "domain": job.get("domain"),
        "synthesis_job_id": job.get("synthesis_job_id"),
        "message": job.get("message"),
        "error": job.get("error"),
    }
    if "score" in job and job.get("score") is not None:
        response["score"] = job["score"]
    return response


@router.get("/{job_id}/visualization")
def get_quality_visualization(
    job_id: str,
    property_name: str = Query(..., description="e.g., 'Column Shapes'"),
    table_name: Optional[str] = Query(default=None),
    column_name: Optional[str] = Query(default=None),
):
    job = QUALITY_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Quality job not completed")

    report = job.get("report")
    if report is None:
        raise HTTPException(
            status_code=400, detail="Quality report not available in memory"
        )

    # Build kwargs with only provided params
    viz_kwargs: Dict[str, Any] = {"property_name": property_name}
    if table_name is not None:
        viz_kwargs["table_name"] = table_name
    if column_name is not None:
        viz_kwargs["column_name"] = column_name

    try:
        fig = report.get_visualization(**viz_kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {str(e)}")

    try:
        fig_json_str = fig.to_json()
        fig_json = json.loads(fig_json_str)
    except Exception:
        # Fallback to to_dict if to_json fails
        try:
            fig_json = fig.to_dict()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to serialize figure: {str(e)}"
            )

    return JSONResponse(content={"figure": fig_json})
