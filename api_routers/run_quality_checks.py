from fastapi import APIRouter, HTTPException, Query, Form
from fastapi.responses import JSONResponse
import uuid, json
from typing import Optional, Dict, Any, Union
from api_routers.utils.shared import (
    QUALITY_JOBS,
    QualityJobRequest,
)
from api_routers.utils.quality_utils import (
    get_completed_synthesis_job_or_raise,
    ensure_zip_file_type_or_raise,
    load_domain_assets_or_raise,
    evaluate_report_and_score,
    persist_quality_job,
    cache_completed_report,
    cache_failed_job,
    get_db_quality_job_or_none,
    build_quality_job_response_from_cache,
    shape_db_quality_job_response,
    build_visualization_kwargs,
    figure_to_serializable_dict,
)

router = APIRouter(prefix="/data_quality_checks", tags=["quality_runs"])


# ---------------------------
# Data Quality routes (evaluate + visualize)
# ---------------------------
@router.post("/run_quality_check")
async def create_quality_job(payload: str = Form(...)):
    try:
        data = json.loads(payload)
        payload_data = QualityJobRequest(**data)
        domain = payload_data.domain
        usecase = payload_data.usecase
        synth_job_id = payload_data.synthesis_job_id

        # Validate usecase - map common variations to valid values
        usecase_mapping = {
            "cohort": "Patient Cohort Builder",
            "patient cohort": "Patient Cohort Builder",
            "patient cohort builder": "Patient Cohort Builder",
            "pharmacovigilance": "Pharmacovigilance",
            "pharma": "Pharmacovigilance",
        }

        if usecase.lower() in usecase_mapping:
            usecase = usecase_mapping[usecase.lower()]

        # Final validation
        from api_routers.utils.shared import AVAILABLE_USECASES

        if usecase not in AVAILABLE_USECASES:
            raise HTTPException(
                status_code=422,
                detail=f"Invalid usecase '{usecase}'. Valid options are: {AVAILABLE_USECASES}",
            )

        synth_job = get_completed_synthesis_job_or_raise(synth_job_id)
        ensure_zip_file_type_or_raise(synth_job.get("file_type"))
        data, metadata = load_domain_assets_or_raise(domain, usecase)

        job_id = str(uuid.uuid4())
        try:
            report, score, property_scores = evaluate_report_and_score(
                real_data=data,
                synthetic_zip_bytes=synth_job.get("csv_bytes"),
                metadata=metadata,
            )
            persist_quality_job(
                job_id=job_id,
                domain=domain,
                synthesis_job_id=synth_job_id,
                status="completed",
                score=score,
                property_scores=property_scores,
                error=None,
                message="Quality evaluation completed",
                report_data=None,
            )
            cache_completed_report(
                job_id=job_id,
                domain=domain,
                synthesis_job_id=synth_job_id,
                score=score,
                report=report,
            )
        except HTTPException:
            raise
        except Exception as e:
            persist_quality_job(
                job_id=job_id,
                domain=domain,
                synthesis_job_id=synth_job_id,
                status="failed",
                score=None,
                error=str(e),
                property_scores=None,
                message=None,
                report_data=None,
            )
            cache_failed_job(job_id, domain, synth_job_id, str(e))

        return {
            "job_id": job_id,
            "status": QUALITY_JOBS[job_id]["status"],
            "score": QUALITY_JOBS[job_id].get("score"),
            "property_scores": property_scores,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{job_id}")
def get_quality_job(job_id: str):
    db_job = get_db_quality_job_or_none(job_id)
    if not db_job:
        return build_quality_job_response_from_cache(job_id)
    return shape_db_quality_job_response(db_job)


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

    viz_kwargs: Dict[str, Any] = build_visualization_kwargs(
        property_name, table_name, column_name
    )

    try:
        fig = report.get_visualization(**viz_kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {str(e)}")

    fig_json = figure_to_serializable_dict(fig)
    return JSONResponse(content={"figure": fig_json})
