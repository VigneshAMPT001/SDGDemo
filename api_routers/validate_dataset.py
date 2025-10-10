"""
Dataset Validation API Endpoints

Provides REST API endpoints for validating custom uploaded datasets.
"""

from fastapi import APIRouter, HTTPException, File, Form, UploadFile
from fastapi.responses import JSONResponse
from typing import List, Dict, Any
import pandas as pd
import io
import os
from api_routers.utils.dataset_validator import (
    validate_custom_dataset,
    ValidationResult,
)
from api_routers.utils.shared import AVAILABLE_DOMAINS, AVAILABLE_USECASES

router = APIRouter(prefix="/validate", tags=["validation"])


@router.post("/dataset")
async def validate_uploaded_dataset(
    files: List[UploadFile] = File(..., description="CSV files to validate"),
    domain: str = Form(..., description="Domain name (e.g., 'Pharma')"),
    usecase: str = Form(
        ..., description="Usecase name (e.g., 'Patient Cohort Builder')"
    ),
):
    """
    Validate uploaded CSV files against domain schema.

    Args:
        files: List of CSV files to validate
        domain: Domain name (e.g., "Pharma")
        usecase: Usecase name (e.g., "Patient Cohort Builder")

    Returns:
        JSON response with validation results
    """
    try:
        # Validate domain and usecase
        if domain not in AVAILABLE_DOMAINS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain '{domain}'. Available domains: {AVAILABLE_DOMAINS}",
            )

        if usecase not in AVAILABLE_USECASES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid usecase '{usecase}'. Available usecases: {AVAILABLE_USECASES}",
            )

        # Read uploaded files
        dataset = await _read_uploaded_files(files)

        if not dataset:
            raise HTTPException(
                status_code=400, detail="No valid CSV files were uploaded"
            )

        # Validate dataset
        validation_result = validate_custom_dataset(dataset, domain, usecase)

        # Convert validation result to JSON-serializable format
        response = _format_validation_result(validation_result)

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/schema/{domain}/{usecase}")
async def get_expected_schema(domain: str, usecase: str):
    """
    Get the expected schema for a domain/usecase combination.

    Args:
        domain: Domain name (e.g., "Pharma")
        usecase: Usecase name (e.g., "Patient Cohort Builder")

    Returns:
        JSON response with expected schema
    """
    try:
        # Validate domain and usecase
        if domain not in AVAILABLE_DOMAINS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid domain '{domain}'. Available domains: {AVAILABLE_DOMAINS}",
            )

        if usecase not in AVAILABLE_USECASES:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid usecase '{usecase}'. Available usecases: {AVAILABLE_USECASES}",
            )

        # Get schema from validator
        from api_routers.utils.dataset_validator import DatasetValidator

        validator = DatasetValidator()
        schema_key = validator._get_schema_key(domain, usecase)

        if not schema_key or schema_key not in validator.domain_schemas:
            raise HTTPException(
                status_code=404,
                detail=f"No schema available for domain '{domain}' and usecase '{usecase}'",
            )

        schema = validator.domain_schemas[schema_key]

        # Format schema for response
        response = {
            "domain": domain,
            "usecase": usecase,
            "required_tables": schema["required_tables"],
            "table_schemas": {},
        }

        for table_name, table_schema in schema["table_schemas"].items():
            response["table_schemas"][table_name] = {
                "required_columns": table_schema["required_columns"],
                "all_columns": list(table_schema["column_types"].keys()),
                "column_types": table_schema["column_types"],
                "primary_key": table_schema["primary_key"],
                "foreign_keys": table_schema["foreign_keys"],
            }

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get schema: {str(e)}")


@router.post("/quick-check")
async def quick_validation_check(
    files: List[UploadFile] = File(..., description="CSV files to quickly validate"),
    domain: str = Form(..., description="Domain name"),
    usecase: str = Form(..., description="Usecase name"),
):
    """
    Perform a quick validation check and return only critical issues.

    Args:
        files: List of CSV files to validate
        domain: Domain name
        usecase: Usecase name

    Returns:
        JSON response with only critical validation issues
    """
    try:
        # Read uploaded files
        dataset = await _read_uploaded_files(files)

        if not dataset:
            return JSONResponse(
                content={
                    "is_valid": False,
                    "critical_issues": ["No valid CSV files were uploaded"],
                    "can_proceed": False,
                }
            )

        # Validate dataset
        validation_result = validate_custom_dataset(dataset, domain, usecase)

        # Extract only critical issues (errors)
        critical_issues = [
            {
                "table": issue.table_name,
                "column": issue.column_name,
                "message": issue.message,
                "suggested_fix": issue.suggested_fix,
            }
            for issue in validation_result.issues
        ]

        response = {
            "is_valid": validation_result.is_valid,
            "critical_issues": critical_issues,
            "can_proceed": len(validation_result.issues) == 0,
            "summary": {
                "total_tables": validation_result.summary["total_tables"],
                "total_issues": validation_result.summary["total_issues"],
                "total_warnings": validation_result.summary["total_warnings"],
            },
        }

        return JSONResponse(content=response)

    except Exception as e:
        return JSONResponse(
            content={
                "is_valid": False,
                "critical_issues": [f"Validation failed: {str(e)}"],
                "can_proceed": False,
            }
        )


async def _read_uploaded_files(files: List[UploadFile]) -> Dict[str, pd.DataFrame]:
    """Read and parse uploaded CSV files."""
    dataset = {}

    for file in files:
        try:
            # Validate file extension
            if not file.filename.lower().endswith(".csv"):
                continue

            # Read file content
            content = await file.read()

            # Parse CSV
            df = pd.read_csv(io.BytesIO(content))

            # Use filename (without extension) as table name
            table_name = os.path.splitext(file.filename)[0]
            dataset[table_name] = df

        except Exception as e:
            # Skip files that can't be parsed
            print(f"Failed to parse file {file.filename}: {e}")
            continue

    return dataset


def _format_validation_result(result: ValidationResult) -> Dict[str, Any]:
    """Convert ValidationResult to JSON-serializable format."""
    return {
        "is_valid": result.is_valid,
        "summary": result.summary,
        "issues": [
            {
                "severity": issue.severity.value,
                "table_name": issue.table_name,
                "column_name": issue.column_name,
                "issue_type": issue.issue_type,
                "message": issue.message,
                "suggested_fix": issue.suggested_fix,
            }
            for issue in result.issues
        ],
        "warnings": [
            {
                "severity": warning.severity.value,
                "table_name": warning.table_name,
                "column_name": warning.column_name,
                "issue_type": warning.issue_type,
                "message": warning.message,
                "suggested_fix": warning.suggested_fix,
            }
            for warning in result.warnings
        ],
        "info": [
            {
                "severity": info.severity.value,
                "table_name": info.table_name,
                "column_name": info.column_name,
                "issue_type": info.issue_type,
                "message": info.message,
                "suggested_fix": info.suggested_fix,
            }
            for info in result.info
        ],
    }
