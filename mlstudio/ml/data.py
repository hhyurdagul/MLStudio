import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)
from streamlit.runtime.uploaded_file_manager import UploadedFile


def read_data(data: UploadedFile | None) -> pl.DataFrame | None:
    return (
        None
        if data is None
        else pl.read_csv(data)
        if data.name.endswith(".csv")
        else pl.read_excel(data)
    )


def get_preprocessing_data(df: pl.DataFrame) -> pl.DataFrame:
    return (
        pl.DataFrame(
            schema={
                "Variable": str,
                "Unique Count": pl.UInt32,
                "Type": str,
                "Preprocessing": str,
            }
        )
        .vstack(
            df.select(pl.selectors.string(include_categorical=True).n_unique())
            .unpivot(variable_name="Variable", value_name="Unique Count")
            .with_columns(
                pl.lit("String").alias("Type"),
                pl.when(pl.col("Unique Count") > 10)
                .then(pl.lit("OrdinalEncoder"))
                .otherwise(pl.lit("OneHotEncoder"))
                .alias("Preprocessing"),
            )
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


def create_transformer_and_transform_data(
    df: pl.DataFrame, preprocessing_df: pl.DataFrame
) -> tuple[ColumnTransformer, pl.DataFrame]:
    transformer = create_preprocessing_transformer(preprocessing_df)
    return transformer, pl.DataFrame(
        transformer.fit_transform(df),
        schema=list(transformer.get_feature_names_out()),
    )
