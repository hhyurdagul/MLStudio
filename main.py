from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
import polars as pl


def read_data(data: UploadedFile | None) -> pl.DataFrame | None:
    return (
        None
        if data is None
        else pl.read_csv(data)
        if data.name.endswith(".csv")
        else pl.read_excel(data)
    )

def render_data_component(df: pl.DataFrame, features: list[str], target: str) -> None:
    with st.expander("Data"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("Loaded Data")
            st.dataframe(df)
        with col2:
            st.write("Selected Training Data")
            st.dataframe(
                df.select(pl.col(features + [target]))
                .to_pandas()
                .style.map(lambda _: "color: darkorange;", subset=[target]),
                hide_index=True,
            )
    

file_selector = st.file_uploader(
    "Upload Train File", type=[".csv", ".xlsx"], accept_multiple_files=False
)

df = read_data(file_selector)
columns = [] if df is None else df.columns

col1, col2 = st.columns(2)
target_selector = col2.selectbox("Select target", columns)
feature_selector = col1.multiselect(
    "Select features", [i for i in columns if i != target_selector]
)

if df is not None:
    render_data_component(df, feature_selector, target_selector)


    
