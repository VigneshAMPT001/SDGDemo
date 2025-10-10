from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

# ---------------------------
# In-memory stores (simple job/config registries)
# ---------------------------
QUALITY_JOBS: Dict[str, Dict[str, Any]] = {}
MODEL_CONFIGS: Dict[str, Any] = {}

# ---------------------------
# Helpers (reused logic from streamlit_app_v2)
# ---------------------------
AVAILABLE_DOMAINS: List[str] = ["Pharma", "BFSI", "Telecom"]
AVAILABLE_USECASES: List[str] = [
    "Patient Cohort Builder",
    "Pharmacovigilance",
]


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
        ..., description="One of: 'SDV (HMA Synthesizer)', 'CTGAN', 'TVAE'"
    )
    domain: str = Field(..., description="Domain name, e.g., 'Pharma'")
    params: Dict[str, Any] = Field(default_factory=dict)
    usecase: str = Field(..., description="Usecase for the seleted domain")


class ModelConfigRequest(BaseModel):
    model_type: str
    config: Dict[str, Any] = Field(default_factory=dict)


class DatasetTable(BaseModel):
    name: str
    rows: int
    columns: int
    column_names: List[str]
    sample_data: List[Dict[str, Any]]
    data_types: Dict[str, str]


class DatasetResponse(BaseModel):
    domain: str
    tables: List[DatasetTable]
    total_tables: int
    metadata: Dict[str, Any]


class QualityJobRequest(BaseModel):
    domain: str = Field(..., description="Domain for real data and metadata")
    usecase: str = Field(..., description="Usecase for the seleted domain")
    synthesis_job_id: str = Field(
        ..., description="Job id from /domains/{domain}/synthesize (SDV multi-table)"
    )


class VisualizationQuery(BaseModel):
    property_name: str = Field(..., description="Report property, e.g. 'Column Shapes'")
    table_name: Optional[str] = Field(default=None)
    column_name: Optional[str] = Field(default=None)
