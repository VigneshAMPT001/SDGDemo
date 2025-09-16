import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
import io
import os
import concurrent.futures
from typing import Optional, Tuple
import asyncio
import threading
import time
from datetime import datetime

# Page configuration
st.set_page_config(page_title="CTGAN Data Synthesizer", page_icon="🤖", layout="wide")


# Reusable executor for background tasks
@st.cache_resource(show_spinner=False)
def get_executor() -> concurrent.futures.ThreadPoolExecutor:
    return concurrent.futures.ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="ctgan"
    )


# Cached asyncio event loop running in a background thread
@st.cache_resource(show_spinner=False)
def get_async_loop() -> asyncio.AbstractEventLoop:
    loop = asyncio.new_event_loop()
    thread = threading.Thread(
        target=loop.run_forever, name="ctgan-async-loop", daemon=True
    )
    thread.start()
    return loop


# --- Initialize session state ---
if "training" not in st.session_state:
    st.session_state.training = False
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "ctgan" not in st.session_state:
    st.session_state.ctgan = None
if "ctgan_future" not in st.session_state:
    st.session_state.ctgan_future = None


# --- Action handlers ---
def start_training():
    st.session_state.training = True
    st.session_state.result = None
    st.session_state.error = None
    st.rerun()  # rerun once to disable buttons immediately


def cancel_training():
    # Best-effort cancel if the task hasn't started
    future = st.session_state.get("ctgan_future")
    if future is not None:
        try:
            future.cancel()
        except Exception:
            pass
        st.session_state.ctgan_future = None
    st.session_state.training = False
    st.session_state.result = None
    st.session_state.error = "Training was cancelled."
    st.rerun()  # rerun once to reset buttons immediately


def train_and_sample(
    data_frame: pd.DataFrame,
    discrete_cols: list,
    epochs: int,
    batch_size: int,
    num_samples: int,
) -> Tuple[pd.DataFrame, dict]:
    ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True)
    ctgan.fit(data_frame, discrete_cols)
    synthetic_df = ctgan.sample(num_samples)
    meta = {
        "epochs": epochs,
        "batch_size": batch_size,
        "num_samples": num_samples,
    }
    return synthetic_df, meta


# Title and description
st.title("🤖 CTGAN Data Synthesizer")
st.markdown(
    "Upload your CSV file to generate synthetic data using Conditional Tabular GAN (CTGAN)"
)

# Sidebar for configuration
st.sidebar.header("Configuration")

# File upload
uploaded_file = st.file_uploader(
    "Choose a CSV file", type="csv", help="Upload a CSV file to generate synthetic data"
)


if uploaded_file is not None:
    try:
        # Read the uploaded CSV
        df = pd.read_csv(uploaded_file)

        # Display file info
        st.success(f"✅ Successfully loaded {uploaded_file.name}")
        st.info(f"📊 Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")

        # Display original data (first 10 rows)
        st.subheader("📋 Original Data (First 10 rows)")
        st.dataframe(df.head(10), width="stretch")

        # Sidebar configuration for CTGAN
        st.sidebar.subheader("CTGAN Parameters")

        epochs = st.sidebar.number_input(
            "Number of Epochs",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of training epochs (higher = better quality but slower)",
        )

        batch_size = st.sidebar.number_input(
            "Batch Size",
            min_value=50,
            max_value=1000,
            value=500,
            help="Number of samples per batch",
        )

        num_samples = st.sidebar.number_input(
            "Number of Synthetic Samples",
            min_value=1,
            max_value=10000,
            value=(len(df)) * 2,
            help="Number of synthetic samples to generate (2x the original by default)",
        )

        st.sidebar.subheader("App Behavior")
        refresh_seconds = st.sidebar.number_input(
            "UI auto-refresh interval (seconds)",
            min_value=2,
            max_value=60,
            value=10,
            step=1,
            help="How often to refresh the UI while training is running",
        )

        st.sidebar.subheader("Discrete Columns")
        st.sidebar.markdown("Select which columns are categorical/discrete:")

        potential_discrete = [
            col
            for col in df.columns
            if df[col].dtype == "object" or df[col].nunique() < 20
        ]

        discrete_columns = st.sidebar.multiselect(
            "Select discrete columns:",
            options=df.columns.tolist(),
            default=potential_discrete,
            help="CTGAN works better when discrete columns are specified",
        )

        # --- UI: Buttons ---
        col1, col2 = st.columns(2)
        with col1:
            st.button(
                "🚀 Generate Synthetic Data",
                on_click=start_training,
                disabled=st.session_state.training,
            )
        with col2:
            st.button(
                "❌ Cancel",
                on_click=cancel_training,
                disabled=not st.session_state.training,
            )

        # If training requested and not already running, start background job via asyncio loop
        if st.session_state.training and st.session_state.ctgan_future is None:
            loop = get_async_loop()
            executor = get_executor()
            st.session_state.ctgan_future = loop.run_in_executor(
                executor,
                train_and_sample,
                df.copy(),
                list(discrete_columns),
                int(epochs),
                int(batch_size),
                int(num_samples),
            )

        # --- Status + Results placeholders ---
        status_placeholder = st.empty()
        result_placeholder = st.empty()

        future = st.session_state.ctgan_future
        if future is not None:
            if not future.done():
                with status_placeholder.container():
                    st.status(
                        "⏳ CTGAN is training...",
                        state="running",
                        expanded=False,
                    )
                # Polling: rerun periodically while training
                time.sleep(float(refresh_seconds))
                st.rerun()
            else:
                try:
                    synthetic_df, meta = future.result()
                    st.session_state.result = (synthetic_df, meta)
                    st.session_state.ctgan_future = None
                    st.session_state.training = False
                    with status_placeholder.container():
                        st.status(
                            "✅ Synthetic data generated successfully!",
                            state="complete",
                            expanded=False,
                        )
                except Exception as e:
                    st.session_state.error = str(e)
                    st.session_state.ctgan_future = None
                    st.session_state.training = False
                    with status_placeholder.container():
                        st.status(f"❌ Error: {e}", state="error", expanded=False)

        # Error display
        if st.session_state.error:
            st.error(f"❌ Error generating synthetic data: {st.session_state.error}")

        # Results display
        if st.session_state.result is not None:
            synthetic_df, meta = st.session_state.result

            with result_placeholder.container():
                st.subheader("🎭 Generated Synthetic Data (First 10 rows)")
                st.dataframe(synthetic_df.head(10), width="stretch")

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("📊 Original Data Statistics")
                    st.write(f"Shape: {df.shape}")
                    st.write(
                        f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                    )
                    if len(df.select_dtypes(include=[np.number]).columns) > 0:
                        st.write("Numeric columns summary:")
                        st.write(df.describe())
                with col2:
                    st.subheader("🎭 Synthetic Data Statistics")
                    st.write(f"Shape: {synthetic_df.shape}")
                    st.write(
                        f"Memory usage: {synthetic_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
                    )
                    if len(synthetic_df.select_dtypes(include=[np.number]).columns) > 0:
                        st.write("Numeric columns summary:")
                        st.write(synthetic_df.describe())

                st.subheader("💾 Download Synthetic Data")
                base_name = os.path.splitext(uploaded_file.name)[0]
                output_filename = f"synthetic_output_{base_name}_{datetime.now()}.csv"
                csv_buffer = io.StringIO()
                df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label=f"📥 Download {output_filename}",
                    data=csv_buffer.getvalue(),
                    file_name=output_filename,
                    mime="text/csv",
                )

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.error("Please make sure you uploaded a valid CSV file.")

else:
    st.info("👆 Please upload a CSV file to get started")
    st.subheader("📝 Example Usage")
    st.markdown(
        """
    1. **Upload a CSV file**  
    2. **Configure CTGAN parameters** in the sidebar  
    3. **Click "Generate Synthetic Data"**  
    4. **Download results** as CSV  
    
    **Tips:**
    - Specify discrete columns correctly  
    - Use more epochs for better quality  
    - Ensure your data has minimal missing values  
    """
    )
