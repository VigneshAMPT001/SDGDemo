from fastapi import APIRouter, HTTPException
from api_routers.route_utils import (
    load_domain_data,
    load_domain_metadata,
)

from api_routers.shared import AVAILABLE_DOMAINS, AVAILABLE_USECASES
from fastapi.responses import StreamingResponse
import io
import zipfile
import json
from datetime import datetime
from api_routers.shared import DatasetTable, DatasetResponse
from fastapi import Query
from typing import Dict, Optional
import pandas as pd
from sdv.metadata import Metadata

router = APIRouter()

router = APIRouter(prefix="/domains", tags=["domains"])


def create_zip_with_metadata(
    tables: Dict[str, pd.DataFrame], metadata: Optional[Metadata], domain: str
) -> bytes:
    """Create a ZIP file containing CSV tables and metadata JSON file."""
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        # Add CSV files
        for table_name, df in tables.items():
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            zip_file.writestr(f"{table_name}.csv", csv_buffer.getvalue())

        # Add metadata JSON file if available
        if metadata:
            try:
                # Convert metadata to JSON string
                metadata_dict = metadata.to_dict()
                metadata_json = json.dumps(metadata_dict, indent=2)
                zip_file.writestr(f"metadata_{domain.lower()}.json", metadata_json)
            except Exception as e:
                # If metadata conversion fails, add a note in the ZIP
                error_note = f"Error including metadata: {str(e)}"
                zip_file.writestr("metadata_error.txt", error_note)

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


# ---------------------------
# Domain routes
# ---------------------------
@router.get("/")
def get_domains():
    return {"domains": AVAILABLE_DOMAINS, "implemented": ["Pharma"]}


@router.get("/usecases")
def get_usecases():
    return {"usecases": AVAILABLE_USECASES}


@router.get("/{domain}/{usecase}/datasets")
def get_domain_datasets(
    domain: str,
    usecase: str,
    format: str = Query("zip", description="Response format: 'zip' or 'csv'"),
):
    if domain not in AVAILABLE_DOMAINS:
        raise HTTPException(status_code=404, detail="Unknown domain")

    data = load_domain_data(domain, usecase)
    if not data:
        raise HTTPException(
            status_code=404, detail="Datasets not available for this domain yet"
        )

    if format.lower() == "csv":
        # Return JSON response with dataset metadata and sample data
        tables = []
        for table_name, df in data.items():
            # Get sample data (first 10 rows)
            sample_data = df.head(10).to_dict(orient="records")

            # Get data types
            data_types = {col: str(dtype) for col, dtype in df.dtypes.items()}

            table_info = DatasetTable(
                name=table_name,
                rows=len(df),
                columns=len(df.columns),
                column_names=list(df.columns),
                sample_data=sample_data,
                data_types=data_types,
            )
            tables.append(table_info)

        # Load metadata for the domain
        metadata = load_domain_metadata(domain, usecase)
        metadata_dict = None
        if metadata:
            try:
                metadata_dict = metadata.to_dict()
            except Exception as e:
                # If metadata conversion fails, include error info
                metadata_dict = {"error": f"Failed to convert metadata: {str(e)}"}

        response = DatasetResponse(
            domain=domain,
            tables=tables,
            total_tables=len(tables),
            metadata=metadata_dict,
        )
        return response
