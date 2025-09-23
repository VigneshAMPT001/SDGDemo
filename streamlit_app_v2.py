import streamlit as st
import pandas as pd
import numpy as np
import json
import io
import os
import zipfile
from pathlib import Path
from typing import Dict, Any, Optional

# SDV imports
from sdv.io.local import CSVHandler
from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer
from sdv.evaluation.multi_table import (
    evaluate_quality,
    get_column_plot,
    get_cardinality_plot,
)

# CTGAN imports
from ctgan import CTGAN, TVAE

# Page configuration
st.set_page_config(
    page_title="DataSyn",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for better styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #667eea;
    }
    .metadata-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Initialize session state
if "synthetic_data" not in st.session_state:
    st.session_state.synthetic_data = None
if "metadata_info" not in st.session_state:
    st.session_state.metadata_info = None
if "synthesizer" not in st.session_state:
    st.session_state.synthesizer = None


def load_domain_data(domain: str) -> Dict[str, pd.DataFrame]:
    """Load data for the selected domain."""
    connector = CSVHandler()

    if domain == "Pharma":
        data = connector.read(
            folder_name="data/pharma",
            file_names=["person.csv", "condition_era.csv"],
            read_csv_parameters={"parse_dates": False, "encoding": "latin-1"},
        )
        return {"person": data["person"], "condition_era": data["condition_era"]}

    # Add more domains here as needed
    return {}


def load_metadata(domain: str) -> Optional[Metadata]:
    """Load metadata for the selected domain."""
    if domain == "Pharma":
        try:
            return Metadata.load_from_json(filepath="metadata/metadata_pharma_v1.json")
        except Exception as e:
            st.error(f"Error loading metadata: {str(e)}")
            return None
    return None


def display_metadata_structure(metadata: Metadata):
    """Display metadata in a structured format."""
    st.subheader("📋 Dataset Metadata")

    metadata_dict = metadata.to_dict()

    # Display tables information
    st.markdown("### 🗂️ Tables")
    for table_name, table_info in metadata_dict.get("tables", {}).items():
        with st.expander(f"📊 {table_name}", expanded=True):
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Columns:**")
                for col_name, col_info in table_info.get("columns", {}).items():
                    sdtype = col_info.get("sdtype", "unknown")
                    st.write(f"• **{col_name}**: {sdtype}")

            with col2:
                st.markdown("**Primary Key:**")
                st.write(f"🔑 {table_info.get('primary_key', 'None')}")

    # Display relationships
    relationships = metadata_dict.get("relationships", [])
    if relationships:
        st.markdown("### 🔗 Relationships")
        for rel in relationships:
            st.write(f"**{rel['parent_table_name']}** → **{rel['child_table_name']}**")
            st.write(f"  Parent Key: {rel['parent_primary_key']}")
            st.write(f"  Foreign Key: {rel['child_foreign_key']}")

    # Display metadata version
    st.markdown("### ℹ️ Metadata Information")
    st.write(f"**Version**: {metadata_dict.get('METADATA_SPEC_VERSION', 'Unknown')}")


def create_zip_download(synthetic_data: Dict[str, pd.DataFrame], domain: str) -> bytes:
    """Create a zip file containing all synthetic datasets."""
    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for table_name, df in synthetic_data.items():
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False, encoding="utf-8")
            zip_file.writestr(f"{table_name}_synthetic.csv", csv_buffer.getvalue())

    zip_buffer.seek(0)
    return zip_buffer.getvalue()


def train_sdv_model(
    data: Dict[str, pd.DataFrame], metadata: Metadata, model_params: Dict[str, Any]
):
    """Train SDV model and generate synthetic data."""
    try:
        # Clean data
        cleaned_data = drop_unknown_references(data, metadata)

        # Initialize synthesizer
        synthesizer = HMASynthesizer(metadata)

        # Fit the model
        with st.spinner("Training SDV model..."):
            synthesizer.fit(cleaned_data)

        # Generate synthetic data
        scale = model_params.get("scale", 1.5)
        synthetic_data = synthesizer.sample(scale=scale)

        # Save synthesizer
        synthesizer.save("synthesizer/synthesizer_pharma_v1.pkl")

        return synthetic_data, synthesizer

    except Exception as e:
        st.error(f"Error training SDV model: {str(e)}")
        return None, None


def train_ctgan_model(data: pd.DataFrame, model_params: Dict[str, Any]):
    """Train CTGAN model and generate synthetic data."""
    try:
        # Initialize CTGAN
        ctgan = CTGAN(
            epochs=model_params.get("epochs", 100),
            batch_size=model_params.get("batch_size", 500),
            verbose=True,
        )

        # Fit the model
        with st.spinner("Training CTGAN model..."):
            ctgan.fit(data, discrete_columns=model_params.get("discrete_columns", []))

        # Generate synthetic data
        num_samples = model_params.get("num_samples", len(data) * 2)
        synthetic_data = ctgan.sample(num_samples)

        return synthetic_data, ctgan

    except Exception as e:
        st.error(f"Error training CTGAN model: {str(e)}")
        return None, None


# Main UI
st.markdown(
    '<h1 class="main-header">🤖 DataSyn - Advanced Data Synthesizer</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "Generate high-quality synthetic data using GAN-based machine learning models"
)

# Sidebar configuration
st.sidebar.header("⚙️ Configuration")

# Model selection
st.sidebar.subheader("🎯 Model Selection")
model_type = st.sidebar.selectbox(
    "Choose Synthesis Model",
    ["SDV (Multi-table)", "CTGAN", "TVAE"],
    help="Select the model type for data synthesis",
)

# Domain selection
st.sidebar.subheader("🏢 Domain Selection")
domain = st.sidebar.selectbox(
    "Choose Domain",
    ["Pharma", "BFSI", "Telecom"],
    help="Select the domain for pre-configured datasets",
)

# Main content area with tabs
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Pre-configured Datasets",
        "📁 Custom Dataset Upload",
        "🏁 Results & Download",
        "🧪 Quality & Visualization",
    ]
)

with tab1:
    st.header("Pre-configured Datasets")

    if st.button("🔄 Load Domain Data", type="primary"):
        with st.spinner(f"Loading {domain} data..."):
            # Load data
            data = load_domain_data(domain)
            metadata = load_metadata(domain)

            if data and metadata:
                st.session_state.domain_data = data
                st.session_state.metadata = metadata
                st.session_state.metadata_info = metadata.to_dict()

                st.success(f"✅ Successfully loaded {domain} dataset!")

                # Display data summary
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Tables", len(data))
                with col2:
                    total_rows = sum(len(df) for df in data.values())
                    st.metric("Total Rows", f"{total_rows:,}")
                with col3:
                    total_cols = sum(len(df.columns) for df in data.values())
                    st.metric("Total Columns", total_cols)

                # Display metadata
                display_metadata_structure(metadata)

                # Display data preview
                st.subheader("📋 Data Preview")
                for table_name, df in data.items():
                    with st.expander(
                        f"📊 {table_name} ({df.shape[0]} rows × {df.shape[1]} columns)"
                    ):
                        st.dataframe(df.head(10), width="stretch")
            else:
                st.error(f"❌ Failed to load {domain} data")

with tab2:
    st.header("Custom Dataset Upload")

    uploaded_files = st.file_uploader(
        "Choose CSV files",
        type="csv",
        accept_multiple_files=True,
        help="Upload one or more CSV files for custom dataset synthesis",
    )

    if uploaded_files:
        st.session_state.custom_data = {}

        for uploaded_file in uploaded_files:
            try:
                df = pd.read_csv(uploaded_file)
                table_name = os.path.splitext(uploaded_file.name)[0]
                st.session_state.custom_data[table_name] = df

                st.success(
                    f"✅ Loaded {uploaded_file.name}: {df.shape[0]} rows × {df.shape[1]} columns"
                )

                with st.expander(f"📊 Preview: {table_name}"):
                    st.dataframe(df.head(10), width="stretch")

            except Exception as e:
                st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")

with tab3:
    st.header("Results & Download")

    # Model parameters
    st.subheader("🎛️ Model Parameters")

    if model_type == "SDV (Multi-table)":
        col1, col2 = st.columns(2)
        with col1:
            scale = st.number_input(
                "Scale Factor", min_value=0.1, max_value=5.0, value=1.5, step=0.1
            )
        with col2:
            st.info(
                "Scale factor determines the size of synthetic data relative to original"
            )

    elif model_type in ["CTGAN", "TVAE"]:
        col1, col2, col3 = st.columns(3)
        with col1:
            epochs = st.number_input("Epochs", min_value=10, max_value=1000, value=100)
        with col2:
            batch_size = st.number_input(
                "Batch Size", min_value=50, max_value=1000, value=500
            )
        with col3:
            num_samples = st.number_input(
                "Synthetic Samples", min_value=100, max_value=50000, value=1000
            )

    # Generate synthetic data button
    if st.button("🚀 Generate Synthetic Data", type="primary"):
        # Determine which data to use
        if "domain_data" in st.session_state and st.session_state.domain_data:
            data = st.session_state.domain_data
            metadata = st.session_state.metadata
            data_source = "domain"
        elif "custom_data" in st.session_state and st.session_state.custom_data:
            data = st.session_state.custom_data
            metadata = None
            data_source = "custom"
        else:
            st.error("❌ Please load data first (either domain data or custom upload)")
            st.stop()

        # Prepare model parameters
        model_params = {}
        if model_type == "SDV (Multi-table)":
            model_params["scale"] = scale
        else:
            model_params["epochs"] = epochs
            model_params["batch_size"] = batch_size
            model_params["num_samples"] = num_samples

            # For CTGAN/TVAE, we need to handle single table
            if len(data) > 1:
                st.warning(
                    "⚠️ CTGAN/TVAE works with single tables. Using the first table."
                )
                data = {list(data.keys())[0]: list(data.values())[0]}

        # Train model and generate synthetic data
        if model_type == "SDV (Multi-table)":
            if metadata is None:
                st.error(
                    "❌ SDV requires metadata. Please use domain data or create metadata for custom data."
                )
                st.stop()

            synthetic_data, synthesizer = train_sdv_model(data, metadata, model_params)
        else:
            # For single table models, use the first (and only) table
            table_name = list(data.keys())[0]
            table_data = data[table_name]

            potential_discrete = []
            for col in table_data.columns:
                if table_data[col].dtype == "object" or table_data[col].nunique() < 20:
                    potential_discrete.append(col)

            model_params["discrete_columns"] = potential_discrete

            if model_type == "CTGAN":
                synthetic_data, synthesizer = train_ctgan_model(
                    table_data, model_params
                )
            else:  # TVAE
                synthetic_data, synthesizer = train_ctgan_model(
                    table_data, model_params
                )  # Same interface

        if synthetic_data is not None:
            st.session_state.synthetic_data = synthetic_data
            st.session_state.synthesizer = synthesizer

            st.success("✅ Synthetic data generated successfully!")

            # Display results
            st.subheader("📊 Generated Synthetic Data")

            if isinstance(synthetic_data, dict):
                # Multi-table results
                for table_name, df in synthetic_data.items():
                    with st.expander(
                        f"📈 {table_name} ({df.shape[0]} rows × {df.shape[1]} columns)"
                    ):
                        st.dataframe(df.head(10), width="stretch")

                        # Statistics
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Original Rows", len(data[table_name]))
                        with col2:
                            st.metric("Synthetic Rows", len(df))
            else:
                # Single table results
                st.dataframe(synthetic_data.head(10), width="stretch")

                # Statistics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Original Rows", len(table_data))
                with col2:
                    st.metric("Synthetic Rows", len(synthetic_data))

    # Download section
    if st.session_state.synthetic_data is not None:
        st.subheader("💾 Download Synthetic Data")

        synthetic_data = st.session_state.synthetic_data

        if isinstance(synthetic_data, dict):
            # Multi-table download
            zip_data = create_zip_download(synthetic_data, domain)

            st.download_button(
                label="📥 Download All Synthetic Data (ZIP)",
                data=zip_data,
                file_name=f"synthetic_data_{domain}_{model_type.lower().replace(' ', '_')}.zip",
                mime="application/zip",
                help="Download all synthetic datasets as a ZIP file",
            )

            # Individual table downloads
            st.markdown("**Individual Table Downloads:**")
            for table_name, df in synthetic_data.items():
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False, encoding="utf-8")

                st.download_button(
                    label=f"📥 Download {table_name}",
                    data=csv_buffer.getvalue(),
                    file_name=f"{table_name}_synthetic.csv",
                    mime="text/csv",
                    key=f"download_{table_name}",
                )
        else:
            # Single table download
            csv_buffer = io.StringIO()
            synthetic_data.to_csv(csv_buffer, index=False, encoding="utf-8")

            st.download_button(
                label="📥 Download Synthetic Data",
                data=csv_buffer.getvalue(),
                file_name=f"synthetic_data_{table_name}_{model_type.lower()}.csv",
                mime="text/csv",
                help="Download the synthetic dataset as CSV",
            )

with tab4:
    st.header("Quality & Visualization")

    if (
        "synthetic_data" in st.session_state
        and isinstance(st.session_state.synthetic_data, dict)
        and "domain_data" in st.session_state
        and "metadata" in st.session_state
        and st.session_state.metadata is not None
    ):
        data = st.session_state.domain_data
        metadata = st.session_state.metadata
        synthetic_data = st.session_state.synthetic_data

        st.subheader("✅ Quality Report")
        try:
            with st.spinner("Evaluating synthetic data quality..."):
                quality_report = evaluate_quality(
                    real_data=data,
                    synthetic_data=synthetic_data,
                    metadata=metadata,
                )
            overall_score = None
            if hasattr(quality_report, "get_score"):
                try:
                    overall_score = quality_report.get_score()
                except Exception:
                    overall_score = None
            if overall_score is not None:
                st.metric("Overall Quality Score", f"{overall_score:.3f}")
            details = None
            if hasattr(quality_report, "get_details"):
                try:
                    details = quality_report.get_details()
                except TypeError:
                    try:
                        details = quality_report.get_details(property_name=None)
                    except Exception:
                        details = None
            if details is not None:
                st.dataframe(details, use_container_width=True)
            else:
                st.write(quality_report)
        except Exception as e:
            st.warning(f"Quality evaluation not available: {str(e)}")

        st.subheader("📊 Column Distribution Comparison")
        try:
            table_options = list(synthetic_data.keys())
            sel_table = st.selectbox("Table", table_options, key="col_plot_table_qv")
            sel_column = st.selectbox(
                "Column",
                list(synthetic_data[sel_table].columns),
                key="col_plot_column_qv",
            )
            if st.button("Show Column Plot", key="show_col_plot_qv"):
                with st.spinner("Generating column plot..."):
                    fig = get_column_plot(
                        real_data=data,
                        synthetic_data=synthetic_data,
                        metadata=metadata,
                        table_name=sel_table,
                        column_name=sel_column,
                    )
                    try:
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception:
                        st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Column plot unavailable: {str(e)}")

        st.subheader("🔗 Cardinality Plot (Parent→Child)")
        try:
            relationships = metadata.to_dict().get("relationships", [])
            if relationships:
                rel_labels = [
                    f"{rel['parent_table_name']} → {rel['child_table_name']}"
                    for rel in relationships
                ]
                sel_idx = st.selectbox(
                    "Relationship",
                    list(range(len(relationships))),
                    format_func=lambda i: rel_labels[i],
                    key="card_plot_rel_qv",
                )
                sel_rel = relationships[sel_idx]
                if st.button("Show Cardinality Plot", key="show_card_plot_qv"):
                    with st.spinner("Generating cardinality plot..."):
                        fig = get_cardinality_plot(
                            real_data=data,
                            synthetic_data=synthetic_data,
                            child_table_name=sel_rel["child_table_name"],
                            parent_table_name=sel_rel["parent_table_name"],
                            child_foreign_key=sel_rel["child_foreign_key"],
                            metadata=metadata,
                        )
                        try:
                            st.plotly_chart(fig, use_container_width=True)
                        except Exception:
                            st.pyplot(fig, use_container_width=True)
            else:
                st.info("No relationships found in metadata.")
        except Exception as e:
            st.warning(f"Cardinality plot unavailable: {str(e)}")
    else:
        st.info(
            "Generate SDV multi-table synthetic data first (using domain data with metadata)."
        )

# # Footer
# st.markdown("---")
# st.markdown(
#     """
#     <div style='text-align: center; color: #666;'>
#         Built with ❤️ using <a href='https://github.com/sdv-dev/SDV'>SDV</a>,
#         <a href='https://github.com/sdv-dev/CTGAN'>CTGAN</a>, and
#         <a href='https://streamlit.io/'>Streamlit</a>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )
