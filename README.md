---
title: SDGDemo
emoji: 📉
colorFrom: indigo
colorTo: pink
sdk: gradio
sdk_version: 5.45.0
app_file: app.py
pinned: false
license: mit
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

## SDGDemo

SDGDemo provides a FastAPI-based service and Streamlit UI for profiling datasets, validating schema/quality, and generating synthetic data using CTGAN/TVAE.

### Key Features
- Data profiling and quality checks (`api_routers/run_quality_checks.py`, `api_routers/utils/*`).
- Dataset validation utilities.
- Synthetic data generation (`api_routers/generate_synthetic_data.py`, `ctgan/*`).
- Optional Azure Blob storage integration (`dbstore/azure_blob.py`).
- Streamlit UI (`streamlit_app_v2.py`).

### Repository Layout
- `app_v2_api.py`: FastAPI application entrypoint (v2 API).
- `api_routers/`: API routes and utilities.
- `ctgan/`: CTGAN/TVAE synthesizers and helpers.
- `metadata/`: Dataset metadata JSONs.
- `data/`: Sample datasets.
- `synthesizer/`: Pre-trained synthesizer artifacts.
- `streamlit_app_v2.py`: Streamlit frontend.

### Prerequisites
- Python 3.12+
- Recommended: virtual environment

### Setup
```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Run the API
```bash
uvicorn app_v2_api:app --host 0.0.0.0 --port 8000 --reload
```
Then open `http://localhost:8000/docs` for interactive Swagger UI.

### Run the Streamlit App
```bash
streamlit run streamlit_app_v2.py
```

### Configuration
Environment variables (optional):
- `AZURE_BLOB_CONNECTION_STRING`: Required for Azure Blob integration.
- `AZURE_BLOB_CONTAINER`: Target container for blob storage.

Place dataset metadata in `metadata/` and sample data in `data/` as needed.

### Testing
```bash
pytest -q
```

### Docker
Build and run with Docker:
```bash
docker build -t sdgdemo:latest .
docker run --rm -p 8000:8000 sdgdemo:latest
```

### Makefile Shortcuts
Common tasks may be available via `Makefile`:
```bash
make help
```

### License
MIT License. See `LICENSE`.

### Contributing
Please read `CONTRIBUTING.rst` and open pull requests against the active branch.
