import polars as pl

from .types import ModelConfig, RowSelection, ValidationStrategy


def validate_labeled_data(
    data: pl.DataFrame,
    features: tuple[str, ...],
    target: str,
) -> None:
    if not features:
        raise ValueError("Select at least one feature.")
    missing = set([*features, target]) - set(data.columns)
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    validate_target(data, target)
    null_columns = [
        column for column in [*features, target] if data[column].null_count() > 0
    ]
    if null_columns:
        raise ValueError(
            "Missing values are not supported yet. Found nulls in: "
            + ", ".join(null_columns)
        )


def validate_target(data: pl.DataFrame, target: str) -> None:
    if not data.schema[target].is_numeric():
        raise ValueError("Regression requires a numeric target.")
    if data[target].null_count() > 0:
        raise ValueError("The target contains missing values.")


def validate_feature_data(
    data: pl.DataFrame,
    features: tuple[str, ...],
    expected_dtypes: dict[str, str],
) -> None:
    missing = [feature for feature in features if feature not in data.columns]
    if missing:
        raise ValueError("Missing required feature columns: " + ", ".join(missing))

    incompatible = [
        feature
        for feature, expected in expected_dtypes.items()
        if _dtype_family(data.schema[feature]) != _dtype_name_family(expected)
    ]
    if incompatible:
        raise ValueError("Incompatible feature types for: " + ", ".join(incompatible))
    null_features = [feature for feature in features if data[feature].null_count() > 0]
    if null_features:
        raise ValueError(
            "Missing values are not supported yet. Found nulls in: "
            + ", ".join(null_features)
        )


def validate_grid_folds(config: ModelConfig, rows: int) -> None:
    if config.use_grid_search and config.cv < 2:
        raise ValueError("Grid-search folds cannot be lower than 2.")
    if config.use_grid_search and config.cv > rows:
        raise ValueError("Grid-search folds cannot exceed the training rows.")


def validate_ordered_training(
    lookback: int | None,
    row_selection: RowSelection,
) -> None:
    validate_lookback(lookback)
    if lookback is not None and row_selection != "Last percent":
        raise ValueError(
            "Target lookback requires training with the last contiguous rows."
        )


def validate_ordered_validation(
    lookback: int | None,
    strategy: ValidationStrategy,
) -> None:
    validate_lookback(lookback)
    if lookback is not None and strategy not in ("Last split", "Cross-validation"):
        raise ValueError(
            "Target lookback requires validation with a chronological last split."
        )


def validate_lookback(lookback: int | None) -> None:
    if lookback is not None and lookback < 1:
        raise ValueError("Lookback must be at least one.")


def _dtype_family(dtype: pl.DataType | type[pl.DataType]) -> str:
    if dtype.is_numeric():
        return "numeric"
    if dtype == pl.Boolean:
        return "boolean"
    if dtype in (pl.String, pl.Categorical, pl.Enum):
        return "string"
    return str(dtype)


def _dtype_name_family(dtype: str) -> str:
    if dtype.startswith(("Int", "UInt", "Float", "Decimal")):
        return "numeric"
    if dtype == "Boolean":
        return "boolean"
    if dtype.startswith(("String", "Categorical", "Enum")):
        return "string"
    return dtype
