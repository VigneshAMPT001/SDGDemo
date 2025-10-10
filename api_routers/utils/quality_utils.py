from tabnanny import verbose
from tokenize import Number
from typing import Any, Dict, Optional, Tuple

from fastapi import HTTPException

from api_routers.utils.route_utils import (
    load_domain_data,
    read_multi_table_zip_bytes,
    load_domain_metadata,
)
from api_routers.utils.shared import QUALITY_JOBS
from dbstore.connect import (
    create_connection,
    get_job_by_id,
    insert_quality_job,
    get_quality_job_by_id,
)


def get_completed_synthesis_job_or_raise(synthesis_job_id: str) -> Dict[str, Any]:
    """Fetch a synthesis job by id and ensure it is completed.

    Raises HTTPException with appropriate status codes on failure.
    """
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        synth_job = get_job_by_id(conn, synthesis_job_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass

    if not synth_job:
        raise HTTPException(status_code=404, detail="Synthesis job not found")
    if synth_job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Synthesis job not completed")
    return synth_job


def ensure_zip_file_type_or_raise(file_type: Optional[str]) -> None:
    if file_type not in ("application/zip", "zip"):
        raise HTTPException(
            status_code=400,
            detail="Quality evaluation currently supports SDV multi-table (zip) results only",
        )


def load_domain_assets_or_raise(
    domain: str, usecase: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    data = load_domain_data(domain, usecase)
    if not data:
        raise HTTPException(status_code=404, detail="Domain data not available")
    metadata = load_domain_metadata(domain, usecase)
    if metadata is None:
        raise HTTPException(
            status_code=400, detail="Metadata required for quality evaluation"
        )
    return data, metadata


def evaluate_report_and_score(
    real_data: Dict[str, Any],
    synthetic_zip_bytes: bytes,
    metadata: Dict[str, Any],
):
    from sdv.evaluation.multi_table import evaluate_quality

    synthetic_data = read_multi_table_zip_bytes(synthetic_zip_bytes)
    report = evaluate_quality(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        verbose=False,
    )
    try:
        # Get the overall score
        score = report.get_score()
        # Ensure the score is a native Python float (avoid numpy scalar types)
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None

        # Get all property scores as a dictionary
        property_scores = {}
        try:
            properties = report.get_properties()
            # properties is a pd.DataFrame, so use .iterrows()
            for _, prop in properties.iterrows():
                prop_name = prop.get("Property")
                prop_score = prop.get("Score")
                if prop_name is not None and prop_score is not None:
                    try:
                        prop_score = float(prop_score)
                    except Exception:
                        prop_score = None
                    property_scores[prop_name] = prop_score
        except Exception:
            property_scores = {}

        print(property_scores)

    except Exception:
        score = None
        property_scores = {}
    return report, score, property_scores


def persist_quality_job(
    job_id: str,
    domain: str,
    synthesis_job_id: str,
    status: str,
    score: Optional[float] = None,
    error: Optional[str] = None,
    message: Optional[str] = None,
    property_scores: Optional[Dict[str, float]] = None,
    report_data: Optional[Dict[str, Any]] = None,
) -> None:
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        # Coerce score to native float to avoid passing numpy types to the DB driver
        if score is not None:
            try:
                score = float(score)
            except Exception:
                score = None
        ok = insert_quality_job(
            conn,
            job_id=job_id,
            domain=domain,
            synthesis_job_id=synthesis_job_id,
            status=status,
            score=score,
            property_scores=property_scores,
            error=error,
            message=message,
            report_data=report_data,
        )
        if not ok:
            raise HTTPException(status_code=500, detail="Failed to persist quality job")
    finally:
        try:
            conn.close()
        except Exception:
            pass


def cache_completed_report(
    job_id: str,
    domain: str,
    synthesis_job_id: str,
    score: Optional[float],
    report: Any,
    message: str = "Quality evaluation completed",
) -> None:
    QUALITY_JOBS[job_id] = {
        "status": "completed",
        "domain": domain,
        "synthesis_job_id": synthesis_job_id,
        "score": score,
        "report": report,
        "message": message,
    }


def cache_failed_job(
    job_id: str, domain: str, synthesis_job_id: str, error: str
) -> None:
    QUALITY_JOBS[job_id] = {
        "status": "failed",
        "domain": domain,
        "synthesis_job_id": synthesis_job_id,
        "error": error,
    }


def build_quality_job_response_from_cache(job_id: str) -> Dict[str, Any]:
    job = QUALITY_JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
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


def get_db_quality_job_or_none(job_id: str) -> Optional[Dict[str, Any]]:
    conn = create_connection()
    if conn is None:
        raise HTTPException(status_code=500, detail="Database connection failed")
    try:
        return get_quality_job_by_id(conn, job_id)
    finally:
        try:
            conn.close()
        except Exception:
            pass


def shape_db_quality_job_response(db_job: Dict[str, Any]) -> Dict[str, Any]:
    response: Dict[str, Any] = {
        "job_id": db_job.get("job_id"),
        "status": db_job.get("status"),
        "domain": db_job.get("domain"),
        "synthesis_job_id": db_job.get("synthesis_job_id"),
        "message": db_job.get("message"),
        "error": db_job.get("error"),
    }
    if db_job.get("score") is not None:
        response["score"] = db_job.get("score")
    return response


def build_visualization_kwargs(
    property_name: str, table_name: Optional[str], column_name: Optional[str]
) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {"property_name": property_name}
    if table_name is not None:
        kwargs["table_name"] = table_name
    if column_name is not None:
        kwargs["column_name"] = column_name
    return kwargs


def figure_to_serializable_dict(fig: Any) -> Dict[str, Any]:
    try:
        fig_json_str = fig.to_json()
        return json.loads(fig_json_str)  # type: ignore[name-defined]
    except Exception:
        try:
            return fig.to_dict()
        except Exception as e:
            raise HTTPException(
                status_code=500, detail=f"Failed to serialize figure: {str(e)}"
            )


# Local import to avoid global dependency when not needed
import json  # noqa: E402
