from streamlit.runtime.uploaded_file_manager import UploadedFile
import streamlit as st
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


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


def create_preprocessing_transformer(
    preprocessing_df: pl.DataFrame,
) -> ColumnTransformer:
    return ColumnTransformer(
        [
            (
                "OneHotEncoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                preprocessing_df.filter(pl.col("Preprocessing") == "OneHotEncoder")[
                    "Variable"
                ].to_list(),
            ),
            (
                "OrdinalEncoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                preprocessing_df.filter(pl.col("Preprocessing") == "OrdinalEncoder")[
                    "Variable"
                ].to_list(),
            ),
            (
                "StandardScaler",
                StandardScaler(),
                preprocessing_df.filter(pl.col("Preprocessing") == "StandardScaler")[
                    "Variable"
                ].to_list(),
            ),
            (
                "MinMaxScaler",
                MinMaxScaler(),
                preprocessing_df.filter(pl.col("Preprocessing") == "MinMaxScaler")[
                    "Variable"
                ].to_list(),
            ),
        ],
        remainder="passthrough",
    )


def get_preprocessing_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.select(pl.selectors.string(include_categorical=True).n_unique())
        .unpivot(variable_name="Variable", value_name="Unique Count")
        .with_columns(
            pl.lit("String").alias("Type"),
            pl.when(pl.col("Unique Count") > 10)
            .then(pl.lit("OrdinalEncoder"))
            .otherwise(pl.lit("OneHotEncoder"))
            .alias("Preprocessing"),
        )
        .vstack(
            df.select(pl.selectors.numeric().n_unique())
            .unpivot(variable_name="Variable", value_name="Unique Count")
            .with_columns(
                pl.lit("Numeric").alias("Type"),
                pl.lit("StandardScaler").alias("Preprocessing"),
            )
        )
    ).select("Variable", "Type", "Unique Count", "Preprocessing")


def render_preprocessing_component(
    df: pl.DataFrame, features: list[str]
) -> ColumnTransformer | None:

    with st.expander("Preprocessing"):
        is_customized = st.checkbox("Customize preprocessing")

        preprocessing_df = get_preprocessing_data(df.select(features))

        if is_customized:
            with st.container(border=True):
                st.write("Customize Encoding")
                st.caption("Click on preprocessing values to make changes")
                _temp_df_1 = st.data_editor(
                    preprocessing_df.filter(pl.col("Type") == "String"),
                    disabled=["Variable", "Type", "Unique Count"],
                    column_order=["Variable", "Preprocessing"],
                    column_config={
                        "Preprocessing": st.column_config.SelectboxColumn(
                            help="Preprocessing type of variable",
                            width="medium",
                            options=["OrdinalEncoder", "OneHotEncoder"],
                            required=True,
                        )
                    },
                )

                st.write("Customize Scaling")
                st.caption("Click on preprocessing values to make changes")
                _temp_df_2 = st.data_editor(
                    preprocessing_df.filter(pl.col("Type") == "Numeric"),
                    disabled=["Variable", "Type", "Unique Count"],
                    column_order=["Variable", "Preprocessing"],
                    column_config={
                        "Preprocessing": st.column_config.SelectboxColumn(
                            help="Preprocessing type of variable",
                            width="medium",
                            options=["StandardScaler", "MinMaxScaler", "None"],
                            required=True,
                        )
                    },
                )

                preprocessing_df = _temp_df_1.vstack(_temp_df_2)

        st.write("Selected Preprocessing Steps")
        st.dataframe(preprocessing_df)

        transformer = create_preprocessing_transformer(preprocessing_df)

        transformed_data = transformer.fit_transform(df.select(pl.col(features)))
        st.write("Transformed Training Data")
        st.dataframe(
            pl.DataFrame(
                transformed_data,
                schema=list(transformer.get_feature_names_out()),
            ),
        )

    return transformer


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
    preprocessing_transformer = render_preprocessing_component(df, feature_selector)
