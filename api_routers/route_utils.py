import os
import zipfile
import io
from typing import Dict, Optional, Any
import pandas as pd
from sdv.io.local import CSVHandler

# SDV imports
from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer
from sdv.utils import load_synthesizer

# CTGAN imports
from ctgan import CTGAN, TVAE
from typing import List


def load_domain_data(domain: str, usecase: str) -> Dict[str, pd.DataFrame]:
    connector = CSVHandler()

    def get_files_in_directory(directory_path: str) -> List[str]:
        files_in_directory = []
        for entry in os.listdir(directory_path):
            full_path = os.path.join(directory_path, entry)
            if os.path.isfile(full_path):
                files_in_directory.append(entry)
        return files_in_directory

    def return_data(
        directory_path: str, files_in_directory: List[str]
    ) -> Dict[str, pd.DataFrame]:
        data = connector.read(
            folder_name=directory_path,
            file_names=files_in_directory,
            read_csv_parameters={"parse_dates": False, "encoding": "latin-1"},
        )
        return {
            file_name.replace(".csv", ""): data[file_name.replace(".csv", "")]
            for file_name in files_in_directory
        }

    if domain == "Pharma":
        if usecase == "Patient Cohort Builder":
            directory_path = "data/pharmacohort"
            files_in_directory = get_files_in_directory(directory_path)
            data = return_data(directory_path, files_in_directory)
            return data

        elif usecase == "Pharmacovigilance":
            directory_path = "data/pharmacv"
            files_in_directory = get_files_in_directory(directory_path)
            data = return_data(directory_path, files_in_directory)
            return data
    return {}


def load_domain_metadata(domain: str, usecase: str) -> Optional[Metadata]:
    """Load or detect metadata for a given domain and usecase.

    First tries to load from existing JSON files, then falls back to detection from data.
    """
    if domain == "Pharma":
        if usecase == "Patient Cohort Builder":
            # Try to load existing metadata file
            metadata_file = "metadata/metadata_pharmacohort_v1.json"
            if os.path.exists(metadata_file):
                try:
                    metadata = Metadata.load_from_json(filepath=metadata_file)
                    if metadata:
                        return metadata
                except Exception as e:
                    print(f"Failed to load metadata from {metadata_file}: {e}")

            # Fallback: detect metadata from data
            data = load_domain_data(domain, usecase)
            if data:
                return Metadata.detect_from_dataframes(data)
            else:
                raise ValueError("No data found for Patient Cohort Builder")

        elif usecase == "Pharmacovigilance":
            # Try to load existing metadata file
            metadata_file = "metadata/metadata_pharmacv_v1.json"
            if os.path.exists(metadata_file):
                try:
                    metadata = Metadata.load_from_json(filepath=metadata_file)
                    if metadata:
                        return metadata
                except Exception as e:
                    print(f"Failed to load metadata from {metadata_file}: {e}")

            # Fallback: detect metadata from data
            data = load_domain_data(domain, usecase)
            if data:
                return Metadata.detect_from_dataframes(data)
            else:
                raise ValueError("No data found for Pharmacovigilance")
    return None


def create_zip_from_tables(tables: Dict[str, pd.DataFrame]) -> bytes:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in tables.items():
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            zip_file.writestr(f"{table_name}.csv", csv_buffer.getvalue())
    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def read_multi_table_zip_bytes(zip_bytes: bytes) -> Dict[str, pd.DataFrame]:
    """Read a ZIP (bytes) containing multiple CSVs into a dict of DataFrames.

    Filenames are expected to be <table_name>.csv.
    """
    tables: Dict[str, pd.DataFrame] = {}
    with zipfile.ZipFile(io.BytesIO(zip_bytes), "r") as zf:
        for name in zf.namelist():
            if not name.lower().endswith(".csv"):
                continue
            table_name = os.path.splitext(os.path.basename(name))[0]
            with zf.open(name) as f:
                df = pd.read_csv(f)
                tables[table_name] = df
    if not tables:
        raise ValueError("ZIP did not contain any CSV files for multi-table data")
    return tables


# ---------------------------
# Training utilities (no Streamlit dependency)
# ---------------------------


def train_sdv_model(
    data: Dict[str, pd.DataFrame], metadata: Metadata, params: Dict[str, Any]
):
    cleaned_data = drop_unknown_references(data, metadata)

    synthesizer = load_synthesizer(filepath="synthesizer/synthesizer_pharma_v1.pkl")
    if synthesizer is None:
        synthesizer = HMASynthesizer(metadata)
        os.makedirs("synthesizer", exist_ok=True)
        synthesizer.save("synthesizer/synthesizer_pharma_v1.pkl")

    synthesizer.fit(cleaned_data)
    scale = params.get("scale", 1.5)
    synthetic_data = synthesizer.sample(scale=scale)

    return synthetic_data, synthesizer


def train_ctgan_or_tvae(model_type: str, table: pd.DataFrame, params: Dict[str, Any]):
    epochs = params.get("epochs", 100)
    batch_size = params.get("batch_size", 500)
    num_samples = params.get("num_samples", max(100, len(table) * 2))

    if model_type == "CTGAN":
        model = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True)
    else:
        model = TVAE(epochs=epochs, batch_size=batch_size, verbose=True)

    # Heuristic for discrete columns
    discrete_columns: List[str] = params.get("discrete_columns") or [
        col
        for col in table.columns
        if table[col].dtype == "object" or table[col].nunique() < 20
    ]

    model.fit(table, discrete_columns=discrete_columns)
    synthetic_df = model.sample(num_samples)
    return synthetic_df, model
