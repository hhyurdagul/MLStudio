import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)


def get_preprocessing_data(df: pl.DataFrame) -> pl.DataFrame:
    supported_columns = set(
        df.select(
            pl.selectors.string(include_categorical=True)
            | pl.selectors.numeric()
            | pl.selectors.boolean()
        ).columns
    )
    unsupported = [column for column in df.columns if column not in supported_columns]
    if unsupported:
        raise ValueError("Unsupported feature types for: " + ", ".join(unsupported))

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
        .vstack(
            df.select(pl.selectors.boolean().n_unique())
            .unpivot(variable_name="Variable", value_name="Unique Count")
            .with_columns(
                pl.lit("Boolean").alias("Type"),
                pl.lit("None").alias("Preprocessing"),
            )
        )
    ).select("Variable", "Type", "Unique Count", "Preprocessing")


def create_preprocessing_transformer(
    preprocessing: pl.DataFrame,
) -> ColumnTransformer:
    def variables(step: str) -> list[str]:
        return preprocessing.filter(pl.col("Preprocessing") == step)[
            "Variable"
        ].to_list()

    return ColumnTransformer(
        [
            (
                "OneHotEncoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                variables("OneHotEncoder"),
            ),
            (
                "OrdinalEncoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                variables("OrdinalEncoder"),
            ),
            ("StandardScaler", StandardScaler(), variables("StandardScaler")),
            ("MinMaxScaler", MinMaxScaler(), variables("MinMaxScaler")),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False,
    )
