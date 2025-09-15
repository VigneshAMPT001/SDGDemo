import streamlit as st
from sdv.io.local import CSVHandler
from sdv.metadata import Metadata
from sdv.utils import drop_unknown_references
from sdv.multi_table import HMASynthesizer

connector = CSVHandler()
data = connector.read(
    folder_name="data/pharma",
    file_names=["person.csv", "condition_era.csv"],
    read_csv_parameters={"parse_dates": False, "encoding": "latin-1"},
)


metadata = Metadata.load_from_json(filepath="metadata_pharma_v1.json")
data = {"person": data["person"], "condition_era": data["condition_era"]}


cleaned_data = drop_unknown_references(data, metadata)

# connector.write(
#     cleaned_data,
#     folder_name="data/pharma_v1_cleaned",
#     to_csv_parameters={"encoding": "latin-1", "index": False},
#     file_name_suffix="_v1",
#     mode="x",
# )


synthesizer = HMASynthesizer(metadata)
synthesizer.fit(cleaned_data)

synthetic_data = synthesizer.sample(scale=1.5)

connector.write(
    synthetic_data,
    folder_name="data/pharma_v1_synthetic",
    to_csv_parameters={"encoding": "latin-1", "index": False},
    file_name_suffix="_v1",
    mode="x",
)

synthesizer.save("synthesizer/synthesizer_pharma_v1.pkl")
