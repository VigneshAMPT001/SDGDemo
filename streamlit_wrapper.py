import streamlit as st
import pandas as pd
import numpy as np
from ctgan import CTGAN
import io
import os

# Page configuration
st.set_page_config(page_title="CTGAN Data Synthesizer", page_icon="🤖", layout="wide")

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
        st.dataframe(df.head(10), use_container_width=True)

        # Sidebar configuration for CTGAN
        st.sidebar.subheader("CTGAN Parameters")

        # Epochs
        epochs = st.sidebar.number_input(
            "Number of Epochs",
            min_value=10,
            max_value=1000,
            value=100,
            help="Number of training epochs (higher = better quality but slower)",
        )

        # Batch size
        batch_size = st.sidebar.number_input(
            "Batch Size",
            min_value=50,
            max_value=1000,
            value=500,
            help="Number of samples per batch",
        )

        # Number of synthetic samples
        num_samples = st.sidebar.number_input(
            "Number of Synthetic Samples",
            min_value=1,
            max_value=10000,
            value=len(df),
            help="Number of synthetic samples to generate",
        )

        # Discrete columns selection
        st.sidebar.subheader("Discrete Columns")
        st.sidebar.markdown("Select which columns are categorical/discrete:")

        # Auto-detect potential discrete columns (string columns or columns with few unique values)
        potential_discrete = []
        for col in df.columns:
            if df[col].dtype == "object" or df[col].nunique() < 20:
                potential_discrete.append(col)

        discrete_columns = st.sidebar.multiselect(
            "Select discrete columns:",
            options=df.columns.tolist(),
            default=potential_discrete,
            help="CTGAN works better when discrete columns are specified",
        )

        # Generate synthetic data button
        if st.button("🚀 Generate Synthetic Data", type="primary"):
            with st.spinner("Training CTGAN model and generating synthetic data..."):
                try:
                    # Initialize CTGAN
                    ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True)

                    # Fit the model
                    ctgan.fit(df, discrete_columns)

                    # Generate synthetic data
                    synthetic_df = ctgan.sample(num_samples)

                    # Display synthetic data
                    st.subheader("🎭 Generated Synthetic Data (First 10 rows)")
                    st.dataframe(synthetic_df.head(10), width="stretch")

                    # Statistics comparison
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
                        if (
                            len(synthetic_df.select_dtypes(include=[np.number]).columns)
                            > 0
                        ):
                            st.write("Numeric columns summary:")
                            st.write(synthetic_df.describe())

                    # Download functionality
                    st.subheader("💾 Download Synthetic Data")

                    # Generate output filename
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    output_filename = f"output_{base_name}.csv"

                    # Convert DataFrame to CSV
                    csv_buffer = io.StringIO()
                    synthetic_df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue()

                    # Download button
                    st.download_button(
                        label=f"📥 Download {output_filename}",
                        data=csv_data,
                        file_name=output_filename,
                        mime="text/csv",
                        help="Click to download the generated synthetic data as CSV",
                    )

                    st.success("✅ Synthetic data generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating synthetic data: {str(e)}")
                    st.error("Please check your data format and try again.")

    except Exception as e:
        st.error(f"❌ Error reading CSV file: {str(e)}")
        st.error("Please make sure you uploaded a valid CSV file.")

else:
    # Show example when no file is uploaded
    st.info("👆 Please upload a CSV file to get started")

    # Show example data
    st.subheader("📝 Example Usage")
    st.markdown(
        """
    1. **Upload a CSV file** using the file uploader above
    2. **Configure CTGAN parameters** in the sidebar:
       - Number of epochs (training iterations)
       - Batch size (samples per batch)
       - Number of synthetic samples to generate
       - Select which columns are discrete/categorical
    3. **Click "Generate Synthetic Data"** to train the model and create synthetic data
    4. **Download the results** as a CSV file
    
    **Tips for better results:**
    - Specify discrete columns correctly (categorical data)
    - Use more epochs for better quality (but slower training)
    - Ensure your data doesn't have too many missing values
    """
    )

# # Footer
# st.markdown("---")
# st.markdown("Built with ❤️ using [CTGAN](https://github.com/sdv-dev/CTGAN) and [Streamlit](https://streamlit.io/)")
