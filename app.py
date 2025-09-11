import gradio as gr
import pandas as pd
import numpy as np
from ctgan import CTGAN
import io
import os

def generate_synthetic_data(file, epochs, batch_size, num_samples, discrete_columns):
    try:
        # Read uploaded file
        df = pd.read_csv(file.name)

        # Train CTGAN
        ctgan = CTGAN(epochs=epochs, batch_size=batch_size, verbose=True)
        ctgan.fit(df, discrete_columns)

        # Generate synthetic samples
        synthetic_df = ctgan.sample(num_samples)

        # Save to CSV
        base_name = os.path.splitext(os.path.basename(file.name))[0]
        output_filename = f"output_{base_name}.csv"
        synthetic_df.to_csv(output_filename, index=False)

        # Return dataframes and file for download
        return (
            df.head(10),
            synthetic_df.head(10),
            synthetic_df.describe(include="all"),
            output_filename
        )

    except Exception as e:
        return f"❌ Error: {str(e)}", None, None, None


# Gradio UI
with gr.Blocks(title="CTGAN Data Synthesizer") as demo:
    gr.Markdown("# 🤖 CTGAN Data Synthesizer")
    gr.Markdown("Upload your CSV file to generate synthetic data using Conditional Tabular GAN (CTGAN).")

    with gr.Row():
        with gr.Column():
            file_input = gr.File(label="Upload CSV", file_types=[".csv"])
            epochs_input = gr.Number(label="Epochs", value=100, precision=0)
            batch_size_input = gr.Number(label="Batch Size", value=500, precision=0)
            num_samples_input = gr.Number(label="Synthetic Samples", value=1000, precision=0)
            discrete_columns_input = gr.Textbox(
                label="Discrete Columns (comma separated)", 
                placeholder="e.g. gender, country"
            )
            run_button = gr.Button("🚀 Generate Synthetic Data")

        with gr.Column():
            original_output = gr.DataFrame(label="📋 Original Data (first 10 rows)")
            synthetic_output = gr.DataFrame(label="🎭 Synthetic Data (first 10 rows)")
            stats_output = gr.DataFrame(label="📊 Synthetic Data Statistics")
            file_output = gr.File(label="💾 Download Synthetic CSV")

    run_button.click(
        fn=lambda file, epochs, batch_size, num_samples, discrete_cols: generate_synthetic_data(
            file,
            int(epochs),
            int(batch_size),
            int(num_samples),
            [c.strip() for c in discrete_cols.split(",")] if discrete_cols else []
        ),
        inputs=[file_input, epochs_input, batch_size_input, num_samples_input, discrete_columns_input],
        outputs=[original_output, synthetic_output, stats_output, file_output]
    )


if __name__ == "__main__":
    demo.launch()
