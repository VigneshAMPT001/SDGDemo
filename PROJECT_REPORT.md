# Synthetic Data Generator (SDGDemo) - Project Report

**Project Name:** Synthetic Data Generator (SDGDemo)  
**Repository:** SDGDemo  
**Owner:** VigneshAMPT001  
**Current Branch:** app_v2_api  
**Report Date:** November 13, 2025

---

## Executive Summary

The **Synthetic Data Generator (SDGDemo)** is an enterprise-grade platform designed to generate privacy-preserving synthetic datasets while maintaining statistical properties of original data. It combines modern deep learning techniques with comprehensive data validation and quality assurance features, addressing critical challenges in data sharing, testing, and compliance.

---

## Project Overview

### What is the Synthetic Data Generator?

SDGDemo is a full-stack application that enables organizations to:

1. **Generate synthetic data** from real datasets using advanced machine learning models
2. **Validate dataset schema and quality** before processing
3. **Profile and analyze data** characteristics
4. **Evaluate synthetic data quality** against real data distributions
5. **Store and retrieve** synthesis jobs and results
6. **Access functionality** through both REST API and interactive UI

### Technology Stack

| Component | Technology |
|-----------|-----------|
| **Backend Framework** | FastAPI (Python) |
| **Frontend UI** | Streamlit |
| **Synthesis Models** | CTGAN, TVAE, SDV (Hierarchical Multi-Table Synthesizer) |
| **Data Processing** | Pandas, NumPy |
| **ML Framework** | PyTorch |
| **Data Quality Evaluation** | SDV (Synthetic Data Vault) |
| **Storage** | SQLite (local), Azure Blob Storage (cloud) |
| **Containerization** | Docker |
| **API Documentation** | Swagger UI (FastAPI built-in) |

### Supported Python Versions

- Python 3.8 through 3.13 (highly flexible version support)

---

## Problem Statement & Pain Points Solved

### Key Pain Points

#### 1. **Data Privacy & Compliance**
- **Challenge:** Organizations cannot share real customer/patient data due to GDPR, HIPAA, and other regulations
- **Solution:** SDGDemo generates statistically representative synthetic data with no private information, enabling safe data sharing

#### 2. **Testing & Development with Production-Like Data**
- **Challenge:** Developers need realistic test data without exposing production systems
- **Solution:** Generate unlimited synthetic datasets that preserve statistical relationships while being completely anonymized

#### 3. **Data Imbalance & Augmentation**
- **Challenge:** Rare disease cohorts or edge cases have insufficient data for model training
- **Solution:** Synthesize additional samples while maintaining underlying distributions and relationships

#### 4. **Data Validation & Quality Assurance**
- **Challenge:** Custom datasets may have schema violations, data quality issues, or missing relationships
- **Solution:** Comprehensive validation framework checks schema compliance, data types, relationships, and data integrity

#### 5. **Lack of Reproducibility in Synthetic Data Generation**
- **Challenge:** Ad-hoc synthetic data generation is inconsistent and hard to track
- **Solution:** Job tracking system with persistent storage allows reproducibility and audit trails

#### 6. **Domain-Specific Data Requirements**
- **Challenge:** Different industries (pharma, healthcare, etc.) have unique data structures
- **Solution:** Pre-configured domains with metadata validation for Patient Cohort and Pharmacovigilance datasets

---

## Core Features & Capabilities

### 1. **Synthetic Data Generation**
**Endpoint:** `/synthesize/*`

Three generation models available:
- **CTGAN (Conditional Tabular GAN):** Best for single-table datasets with complex relationships
- **TVAE (Tabular Variational AutoEncoder):** Effective for capturing column distributions and dependencies
- **SDV HMA Synthesizer:** Multi-table hierarchical synthesis maintaining referential integrity

**Key Capabilities:**
- Single and multi-table synthesis
- Support for multiple data types (numerical, categorical, datetime)
- Foreign key relationship preservation
- Configurable training epochs and batch sizes
- Job persistence with UUID tracking

### 2. **Data Quality Evaluation**
**Endpoint:** `/data_quality_checks/*`

Quality assessment metrics:
- **Column Shapes:** Compares statistical distributions (histograms, KDE plots)
- **Cardinality:** Validates count of unique values
- **Intertable Trends:** Checks foreign key relationships are preserved
- **Column Pair Trends:** Ensures correlations between columns match original data
- **Statistical Scores:** Provides quantitative quality metrics

**Visualization Support:**
- Interactive plots for dimension analysis
- Comparison charts: Real vs. Synthetic
- Trend analysis and anomaly detection

### 3. **Dataset Validation**
**Endpoint:** `/validate/*`

Comprehensive validation checks:
- **Schema Compliance:** Ensures required tables and columns exist
- **Data Type Validation:** Verifies column types match expectations
- **Referential Integrity:** Validates foreign key relationships
- **Data Quality Rules:** Detects duplicate primary keys, orphaned references
- **Custom Rules:** Domain-specific validation rules

### 4. **Data Profiling & Analysis**
**Endpoint:** `/data_profiler/*`

Statistical profiling features:
- **Numeric Statistics:** Mean, median, mode, std dev, percentiles, skewness, kurtosis
- **Categorical Analysis:** Value counts, unique values, cardinality
- **Missing Data Detection:** Identifies null values and gaps
- **Correlation Analysis:** Computes Pearson and Spearman correlations
- **Distribution Fitting:** Tests for normality and fitting probability distributions
- **Schema Inference:** Auto-detects data types from samples

### 5. **Model Management**
**Endpoint:** `/models/*`

Model lifecycle management:
- Create and configure models with custom parameters
- Train models on datasets
- Store trained artifacts for reuse
- Model versioning and metadata tracking

### 6. **Domain Management**
**Endpoint:** `/domains/*`

Pre-configured industry domains:
- **Pharma Domain**
  - **Patient Cohort Builder:** OMOP-compatible patient demographic data
  - **Pharmacovigilance:** Adverse reaction, drug exposure, therapeutic indication data
- Extensible architecture for adding new domains

---

## Architecture & Project Structure

```
SDGDemo/
├── app_v2_api.py                 # FastAPI v2 application entrypoint
├── streamlit_app_v2.py           # Streamlit UI frontend
├── main.py                        # Legacy API implementation
├── pyproject.toml               # Project metadata & dependencies
├── requirements.txt             # Python package dependencies
├── Dockerfile                   # Container image definition
│
├── api_routers/                 # FastAPI route modules
│   ├── generate_synthetic_data.py    # Synthesis endpoints
│   ├── run_quality_checks.py        # Quality evaluation endpoints
│   ├── validate_dataset.py          # Dataset validation endpoints
│   ├── data_profiler.py             # Data profiling endpoints
│   ├── domains.py                   # Domain management
│   ├── models.py                    # Model lifecycle management
│   └── utils/                       # Shared utilities
│       ├── shared.py                # Common data models & configs
│       ├── route_utils.py           # Routing helper functions
│       ├── quality_utils.py         # Quality check utilities
│       ├── dataset_validator.py     # Validation logic
│       └── shared.py                # Shared configurations
│
├── ctgan/                       # CTGAN/TVAE synthesizer module
│   ├── synthesizers/
│   │   ├── base.py
│   │   ├── ctgan.py
│   │   └── tvae.py
│   ├── data_sampler.py
│   ├── data_transformer.py
│   ├── data.py
│   └── demo.py
│
├── dbstore/                     # Database connectivity
│   ├── connect.py               # Connection management
│   ├── azure_blob.py            # Azure Blob Storage integration
│   └── __init__.py
│
├── data/                        # Sample datasets
│   ├── pharmacohort/            # OMOP patient cohort data
│   │   ├── person.csv
│   │   ├── condition_occurrence.csv
│   │   ├── drug_exposure.csv
│   │   └── procedure_occurrence.csv
│   └── pharmacv/                # Pharmacovigilance data
│       ├── DEMO_cleaned.csv
│       ├── DRUG_cleaned.csv
│       ├── INDI_cleaned.csv
│       ├── OUTC_cleaned.csv
│       ├── REAC_cleaned.csv
│       ├── RPSR_cleaned.csv
│       └── THER_cleaned.csv
│
├── metadata/                    # Dataset metadata definitions
│   ├── Pharma_Patient Cohort Builder_v1.json
│   └── Pharma_Pharmacovigilance_v1.json
│
├── synData/                     # Generated synthetic data outputs
│   └── pharma/
│
├── quality_reports/             # Quality evaluation report storage
│
├── synthesizer/                 # Pre-trained synthesizer models
│
└── README.md, DATASET_VALIDATION.md, LICENSE
```

---

## Use Cases

### 1. **Healthcare & Pharmaceutical Research**
- Generate synthetic patient cohorts for clinical trials without exposing PHI
- Create realistic pharmacovigilance datasets for adverse event research
- Share data with external research partners under compliance requirements

### 2. **Software Testing & QA**
- Populate test environments with production-like data structures
- Test data pipelines and ETL processes safely
- Validate edge cases with synthetic variations

### 3. **Machine Learning Model Development**
- Generate training data for model validation
- Augment imbalanced datasets (rare diseases, edge cases)
- Create privacy-safe datasets for demonstrations and competitions

### 4. **Data Science Education**
- Provide realistic datasets for training programs
- Share complex multi-table relationships for learning analytics techniques
- Enable hands-on labs without privacy concerns

### 5. **Compliance & Auditing**
- Demonstrate data governance capabilities with synthetic datasets
- Create audit trails of data generation and usage
- Maintain reproducible data processing pipelines

### 6. **API/Integration Testing**
- Generate complex multi-table datasets for API testing
- Create edge cases and boundary conditions systematically
- Validate system performance with large synthetic datasets

---

## API Endpoints Summary

### Synthesis Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/synthesize/train` | Train and generate synthetic data |
| GET | `/synthesize/job/{job_id}` | Retrieve synthesis job results |
| GET | `/synthesize/jobs` | List all synthesis jobs |

### Quality Check Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/data_quality_checks/run_quality_check` | Execute quality evaluation |
| POST | `/data_quality_checks/visualize` | Generate visualization plots |
| GET | `/data_quality_checks/job/{job_id}` | Retrieve quality report |

### Validation Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/validate/dataset` | Validate uploaded CSV files |
| GET | `/validate/schema/{domain}/{usecase}` | Get expected schema |

### Data Profiler Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/data_profiler/profile` | Profile dataset characteristics |
| POST | `/data_profiler/basic_stats` | Generate basic statistics |
| POST | `/data_profiler/correlation` | Compute correlation matrix |

### Model Management Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/models/create` | Create new model config |
| GET | `/models/list` | List available models |

### Domain Management Endpoints
| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/domains/list` | List supported domains |
| GET | `/domains/{domain}/usecases` | Get usecases for domain |

---

## Key Technologies & Integrations

### Machine Learning & Data Science
- **CTGAN/TVAE:** State-of-the-art tabular synthetic data generation
- **SDV (Synthetic Data Vault):** Multi-table synthesis with relationship preservation
- **PyTorch:** Deep learning backend for neural network models
- **Pandas/NumPy:** Data manipulation and numerical computing

### Web Framework & APIs
- **FastAPI:** Modern async Python web framework with automatic API docs
- **Streamlit:** Rapid UI development for interactive dashboards
- **Swagger/OpenAPI:** Interactive API documentation

### Data Storage & Management
- **SQLite:** Embedded database for job tracking (development)
- **Azure Blob Storage:** Cloud-based file storage for datasets and reports
- **CSV Format:** Standard data interchange format for datasets

### Infrastructure & Deployment
- **Docker:** Containerization for consistent deployments
- **Python Multiprocessing:** Parallel data processing
- **Environment Variables:** Configuration management

---

## Supported Domains & Usecases

### Pharma Domain

#### Patient Cohort Builder (OMOP CDM)
**Purpose:** Healthcare research with patient demographics, conditions, procedures, and drug exposures

**Required Tables:**
- `person.csv` - Patient demographics (ID, age, gender, etc.)
- `condition_occurrence.csv` - Clinical diagnoses and conditions
- `drug_exposure.csv` - Medication administration records
- `procedure_occurrence.csv` - Clinical procedures and treatments

**Relationships:**
- Foreign key: `condition_occurrence.person_id` → `person.person_id`
- Foreign key: `drug_exposure.person_id` → `person.person_id`
- Foreign key: `procedure_occurrence.person_id` → `person.person_id`

#### Pharmacovigilance
**Purpose:** Adverse drug reaction monitoring and safety analysis

**Required Tables:**
- `DEMO.csv` - Demographic information
- `DRUG.csv` - Drug administration data
- `INDI.csv` - Medical indications
- `OUTC.csv` - Patient outcomes
- `REAC.csv` - Adverse reactions reported
- `RPSR.csv` - Report suspect relationships
- `THER.csv` - Therapeutic areas and classifications

---

## Configuration & Deployment

### Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# Run API
uvicorn app_v2_api:app --host 0.0.0.0 --port 8000 --reload

# Run Streamlit UI
streamlit run streamlit_app_v2.py
```

### Docker Deployment
```bash
docker build -t sdgdemo:latest .
docker run --rm -p 8000:8000 sdgdemo:latest
```

### Environment Variables
- `AZURE_BLOB_CONNECTION_STRING`: Connection string for Azure Blob Storage
- `AZURE_BLOB_CONTAINER`: Target container name for blob operations
- `PORT`: Server port (default: 8000 for API, 7860 for Streamlit)

---

## Data Flow Diagram

```
User Input (CSV/JSON)
        ↓
┌─────────────────────────────┐
│  Dataset Validation Layer   │ ← Schema & quality checks
│  (/validate/dataset)        │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│  Data Profiling Layer       │ ← Statistical analysis
│  (/data_profiler/profile)   │
└──────────┬──────────────────┘
           ↓
┌─────────────────────────────┐
│  Synthesis Model Training   │ ← CTGAN/TVAE/HMA
│  (/synthesize/train)        │
└──────────┬──────────────────┘
           ↓
Synthetic Data Generated
        ↓
┌─────────────────────────────┐
│  Quality Evaluation         │ ← Distributions, correlations
│  (/data_quality_checks/run) │
└──────────┬──────────────────┘
           ↓
Quality Report + Visualizations
        ↓
┌─────────────────────────────┐
│  Job Persistence Store      │ ← SQLite/Azure Blob
│  (Retrieval & Audit)        │
└─────────────────────────────┘
```

---

## Key Features at a Glance

| Feature | Benefit |
|---------|---------|
| **Multi-Model Support** | Choose best algorithm (CTGAN, TVAE, HMA) for dataset |
| **Relationship Preservation** | Maintains foreign keys & data dependencies |
| **Quality Metrics** | Validate synthetic data matches real distributions |
| **Job Persistence** | Track, retrieve, and audit all synthesis jobs |
| **Schema Validation** | Catch data issues before processing |
| **Multi-Domain Support** | Pre-configured pharma, extensible to other domains |
| **Cloud Integration** | Azure Blob Storage for scalable data management |
| **Dual Interface** | REST API for programmatic access, Streamlit UI for exploration |
| **Interactive Visualizations** | Compare real vs synthetic data visually |
| **Docker Ready** | Easy containerization and deployment |

---

## Technical Strengths

1. **Privacy-First Design:** Synthetic data generation ensures no real PII is exposed
2. **Statistically Sound:** Maintains distributions, correlations, and relationships
3. **Scalable Architecture:** FastAPI's async design handles concurrent requests
4. **Extensible:** Modular API router system allows adding new domains and features
5. **Production-Ready:** Job tracking, error handling, and comprehensive logging
6. **Well-Documented:** Swagger/OpenAPI docs, markdown guides, and inline code documentation
7. **Flexible Deployment:** Supports local development, Docker, and cloud platforms

---

## Extensibility & Future Enhancements

### Potential Enhancements
1. **Additional Domains:** Fintech, e-commerce, IoT/manufacturing
2. **Advanced Models:** Diffusion models, Generative Pre-trained Transformers (GPT)
3. **Differential Privacy:** Add formal privacy guarantees with DP-SGD
4. **Real-time Synthesis:** Streaming data generation for live applications
5. **ML Pipeline Integration:** Scikit-learn, TensorFlow Keras model preservation
6. **Advanced Analytics:** Anomaly detection, clustering analysis
7. **Federated Learning:** Train models across distributed data sources
8. **Web UI Enhancements:** Real-time job monitoring, advanced filtering

---

## Conclusion

The **Synthetic Data Generator (SDGDemo)** addresses critical challenges in data privacy, compliance, and testing by providing an enterprise-grade platform for generating realistic, privacy-preserving synthetic datasets. With support for multiple generation algorithms, comprehensive quality evaluation, and domain-specific validation, it empowers organizations to securely share and utilize data while maintaining statistical integrity.

The flexible architecture and dual interface (REST API + Streamlit UI) make it accessible to both technical and non-technical users, enabling widespread adoption across research, development, and compliance teams.

---

## Contact & Repository

- **Repository:** https://github.com/VigneshAMPT001/SDGDemo
- **Current Branch:** app_v2_api
- **Documentation:** See README.md, DATASET_VALIDATION.md

---

*Report Generated: November 13, 2025*
