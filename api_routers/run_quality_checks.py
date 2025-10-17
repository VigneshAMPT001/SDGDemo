import pickle
from api_routers.utils.route_utils import read_multi_table_zip_bytes
from fastapi import APIRouter, HTTPException, Form
from fastapi.responses import JSONResponse
import uuid, json
from typing import Optional, Dict, Any
from api_routers.utils.shared import (
    QUALITY_JOBS,
    QualityJobRequest,
)
import os
from api_routers.utils.quality_utils import (
    generate_quality_report,
    get_completed_synthesis_job_or_raise,
    ensure_zip_file_type_or_raise,
    get_report_scores,
    load_domain_assets_or_raise,
    persist_quality_job,
    cache_completed_report,
    cache_failed_job,
    get_db_quality_job_or_none,
    build_quality_job_response_from_cache,
    shape_db_quality_job_response,
    build_visualization_kwargs,
    figure_to_serializable_dict,
)

VIZ_PROPERTIES = [
    "Column Shapes",
    "Cardinality",
    "Intertable Trends",
    "Column Pair Trends",
]

# Validate usecase - map common variations to valid values
USECASE_MAPPING = {
    "cohort": "Patient Cohort Builder",
    "patient cohort": "Patient Cohort Builder",
    "patient cohort builder": "Patient Cohort Builder",
    "pharmacovigilance": "Pharmacovigilance",
    "pharma": "Pharmacovigilance",
}

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

        if usecase.lower() in USECASE_MAPPING:
            usecase = USECASE_MAPPING[usecase.lower()]

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
            report_path = generate_quality_report(
                real_data=data,
                synthetic_zip_bytes=synth_job.get("csv_bytes"),
                metadata=metadata,
                synth_iden=synth_job_id,
            )

            score, property_scores = get_report_scores(report_path)

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
                property_scores=property_scores,
                report=None,
            )
            return {
                "job_id": job_id,
                "status": QUALITY_JOBS[job_id]["status"],
                "score": QUALITY_JOBS[job_id].get("score"),
                "property_scores": QUALITY_JOBS[job_id].get("property_scores"),
            }
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
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/{synth_job_id}")
def get_quality_job(synth_job_id: str):
    db_job = get_db_quality_job_or_none(synth_job_id)
    if not db_job:
        return build_quality_job_response_from_cache(synth_job_id)
    return shape_db_quality_job_response(db_job)


@router.post("/getVizArgs")
def get_viz(job_id: str = Form(...)):
    db_job = get_completed_synthesis_job_or_raise(job_id)
    if not db_job:
        raise HTTPException(status_code=404, detail="Job not found")
    if db_job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Synthesis job not completed")
    csv_bytes = db_job.get("csv_bytes")
    if not csv_bytes:
        raise HTTPException(status_code=400, detail="Synthesis Data not available")

    if db_job.get("file_type") == "application/zip":
        synth_data_tables = read_multi_table_zip_bytes(csv_bytes)
        table_columns_pair = {}

        for table, df in synth_data_tables.items():
            table_columns_pair[table] = df.columns.tolist()

        return JSONResponse(
            content={"vizKwargs": table_columns_pair, "properties": VIZ_PROPERTIES}
        )


@router.post("/get_visualizations")
def get_quality_visualization(
    synth_job_id: str = Form(...),
    property_name: str = Form(..., description="e.g., 'Column Shapes'"),
    table_name: Optional[str] = Form(default=None),
    column_name: Optional[str] = Form(default=None),
):
    report_path = f"quality_reports/{synth_job_id}.pkl"
    job = get_db_quality_job_or_none(synth_job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Quality job not completed")

    if os.path.exists(report_path):
        from sdmetrics.reports.multi_table import QualityReport

        report = QualityReport.load(report_path)

        if report is None:
            raise HTTPException(status_code=400, detail="Quality report not available")

        viz_kwargs: Dict[str, Any] = build_visualization_kwargs(
            property_name, table_name
        )

    try:
        fig = report.get_visualization(**viz_kwargs)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Visualization error: {str(e)}")

    fig_json = figure_to_serializable_dict(fig)
    return JSONResponse(content={"figure": fig_json})
