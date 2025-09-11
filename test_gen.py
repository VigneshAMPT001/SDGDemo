# from ctgan import CTGAN
# from ctgan import load_demo

# real_data = load_demo()

# # Names of the columns that are discrete
# discrete_columns = [
#     "workclass",
#     "education",
#     "marital-status",
# ]

# ctgan = CTGAN(epochs=10)
# ctgan.fit(real_data, discrete_columns)

# # Create synthetic data
# synthetic_data = ctgan.sample(10)

import pandas as pd
from sdv.single_table import GaussianCopulaSynthesizer
from sdv.metadata import Metadata

DEMO_URL = "condition_era.csv"
data = pd.read_csv(DEMO_URL)
metadata = Metadata.detect_from_dataframe(data)

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data)
synthetic_data = synthesizer.sample(num_rows=100)

synthetic_data.to_csv("output_condition_era.csv", index=False)

# import pandas as pd
# from ctgan import CTGAN
# import os

# # Test with existing CSV files
# csv_files = ["care_site.csv", "condition_era.csv"]

# for csv_file in csv_files:
#     if os.path.exists(csv_file):
#         print(f"Testing with {csv_file}...")
#         try:
#             df = pd.read_csv(csv_file)
#             print(f"  Shape: {df.shape}")
#             print(f"  Columns: {list(df.columns)}")

#             # Auto-detect discrete columns
#             discrete_cols = []
#             for col in df.columns:
#                 if df[col].dtype == "object" or df[col].nunique() < 20:
#                     discrete_cols.append(col)
#             print(f"  Discrete columns: {discrete_cols}")

#             # Quick CTGAN test
#             ctgan = CTGAN(epochs=1, verbose=False)
#             ctgan.fit(df, discrete_cols)
#             synthetic = ctgan.sample(5)
#             print(f"  Generated synthetic data shape: {synthetic.shape}")
#             print("  ✅ Success!")
#         except Exception as e:
#             print(f"  ❌ Error: {e}")
#         print()
