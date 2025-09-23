from fastapi import APIRouter
from api_routers.shared import MODEL_CONFIGS, ModelConfigRequest

router = APIRouter()

router = APIRouter(prefix="/models", tags=["models"])


# ---------------------------
# Model routes
# ---------------------------
@router.get("/")
def get_models():
    return {
        "models": [
            {"id": "sdv", "name": "SDV (Multi-table)", "capability": "multi-table"},
            {"id": "ctgan", "name": "CTGAN", "capability": "single-table"},
            {"id": "tvae", "name": "TVAE", "capability": "single-table"},
        ]
    }


@router.post("/config")
def post_model_config(cfg: ModelConfigRequest):
    MODEL_CONFIGS[cfg.model_type] = cfg.config
    return {"model_type": cfg.model_type, "saved": True}
