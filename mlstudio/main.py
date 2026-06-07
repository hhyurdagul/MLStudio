from __future__ import annotations

from typing import cast

import altair as alt
import polars as pl
import streamlit as st
from sklearn.model_selection import GridSearchCV

from mlstudio.components import (
    render_data_preview,
    render_dataset_selector,
    render_feature_target_selector,
    render_modeling_component,
    render_preprocessing_component,
)
from mlstudio.ml.data import (
    create_preprocessing_transformer,
    get_preprocessing_data,
    read_data,
)
from mlstudio.ml.modeling import (
    CrossValidationMetrics,
    RegressionMetrics,
    RowSelection,
    ValidationStrategy,
    calculate_metrics,
    create_artifact,
    create_estimator,
    cross_validate_estimator,
    deserialize_artifact,
    prediction_frame,
    select_training_rows,
    serialize_artifact,
    split_validation_data,
    validate_feature_columns,
    validate_feature_schema,
)


def _validate_training_columns(
    df: pl.DataFrame,
    features: list[str],
    target: str,
) -> None:
    if not features:
        raise ValueError("Select at least one feature.")
    if not df.schema[target].is_numeric():
        raise ValueError("Regression requires a numeric target.")
    columns = features + [target]
    null_columns = [column for column in columns if df[column].null_count() > 0]
    if null_columns:
        raise ValueError(
            "Missing values are not supported yet. Found nulls in: "
            + ", ".join(null_columns)
        )


def _render_preprocessing(
    df: pl.DataFrame,
    features: list[str],
    *,
    key_prefix: str,
):
    preprocessing_df = get_preprocessing_data(df.select(features))
    return render_preprocessing_component(
        preprocessing_df,
        key_prefix=key_prefix,
    )


def _show_metrics(metrics: RegressionMetrics) -> None:
    columns = st.columns(4)
    columns[0].metric("R²", f"{metrics.r2:.4f}")
    columns[1].metric("MAE", f"{metrics.mae:.4f}")
    columns[2].metric("RMSE", f"{metrics.rmse:.4f}")
    columns[3].metric(
        "MAPE",
        "N/A" if metrics.mape is None else f"{metrics.mape:.2f}%",
    )
    if metrics.mape is None:
        st.caption("MAPE is unavailable because every actual target value is zero.")


def _show_cross_validation_metrics(metrics: CrossValidationMetrics) -> None:
    columns = st.columns(4)
    columns[0].metric("R²", f"{metrics.r2_mean:.4f} ± {metrics.r2_std:.4f}")
    columns[1].metric("MAE", f"{metrics.mae_mean:.4f} ± {metrics.mae_std:.4f}")
    columns[2].metric(
        "RMSE",
        f"{metrics.rmse_mean:.4f} ± {metrics.rmse_std:.4f}",
    )
    columns[3].metric(
        "MAPE",
        (
            "N/A"
            if metrics.mape_mean is None or metrics.mape_std is None
            else f"{metrics.mape_mean:.2f}% ± {metrics.mape_std:.2f}%"
        ),
    )


def _show_predictions(results: pl.DataFrame, *, key: str) -> None:
    data_tab, graph_tab = st.tabs(["Data", "Graph"])
    with data_tab:
        st.dataframe(results, hide_index=True)
        st.download_button(
            "Download predictions",
            data=results.write_csv().encode(),
            file_name="predictions.csv",
            mime="text/csv",
            key=key,
            on_click="ignore",
        )
    with graph_tab:
        chart_data = (
            results.with_row_index("Row")
            .unpivot(
                index="Row",
                variable_name="Series",
                value_name="Value",
            )
            .to_pandas()
        )
        chart = (
            alt.Chart(chart_data)
            .mark_line()
            .encode(
                x=alt.X("Row:Q", title="Row"),
                y=alt.Y("Value:Q", title="Value"),
                color=alt.Color(
                    "Series:N",
                    scale=alt.Scale(
                        domain=["Real", "Prediction"],
                        range=["#1f77b4", "#ff7f0e"],
                    ),
                    legend=alt.Legend(title=None),
                ),
                tooltip=["Row:Q", "Series:N", "Value:Q"],
            )
        )
        st.altair_chart(chart, width="stretch")


def _show_grid_results(estimator: object) -> None:
    if isinstance(estimator, GridSearchCV):
        st.write("Best grid-search parameters")
        st.json(estimator.best_params_)
        st.metric("Best cross-validation R²", f"{estimator.best_score_:.4f}")


def _render_training_mode() -> None:
    st.header("Training")
    training_file_column, test_file_column = st.columns(2)
    with training_file_column:
        train_df = render_dataset_selector(
            "Upload training data",
            read_data,
            key="training_data",
        )
    with test_file_column:
        test_df = render_dataset_selector(
            "Upload test data",
            read_data,
            key="training_test_data",
        )
    if train_df is None:
        st.info("Upload training data to configure a model.")
        return

    features, target = render_feature_target_selector(
        train_df,
        key_prefix="training",
    )
    with st.expander("Data preview"):
        render_data_preview(train_df, features, target)
    if not features:
        st.info("Select at least one feature to configure preprocessing and modeling.")
        return

    selection_column, percent_column = st.columns(2)
    row_method = selection_column.selectbox(
        "Training rows",
        ["Random percent", "Last percent"],
        key="training_row_method",
    )
    train_percent = percent_column.slider(
        "Percent used for training",
        1,
        100,
        100,
        key="training_percent",
    )

    try:
        preprocessing_df = _render_preprocessing(
            train_df,
            features,
            key_prefix="training",
        )
    except ValueError as error:
        st.error(str(error))
        return

    model_config, model_config_is_valid = render_modeling_component(
        key_prefix="training"
    )
    if not st.button(
        f"Train {model_config.definition.label}",
        type="primary",
        disabled=not model_config_is_valid,
        key="train_model",
    ):
        return

    try:
        _validate_training_columns(train_df, features, target)
        selected_train_df = select_training_rows(
            train_df,
            cast(RowSelection, row_method),
            train_percent,
        )
        if model_config.use_grid_search and model_config.cv > selected_train_df.height:
            raise ValueError(
                "Grid-search folds cannot exceed the selected training rows."
            )
        transformer = create_preprocessing_transformer(preprocessing_df)
        estimator = create_estimator(transformer, model_config)
        with st.spinner("Training model..."):
            estimator.fit(
                selected_train_df.select(features),
                selected_train_df[target],
            )

        st.success(f"Trained on {selected_train_df.height} rows.")
        _show_grid_results(estimator)
        artifact = create_artifact(
            estimator,
            train_df,
            features,
            target,
            model_config.definition.label,
        )
        st.download_button(
            "Download model bundle",
            data=serialize_artifact(artifact),
            file_name="mlstudio-model.joblib",
            mime="application/octet-stream",
            key="download_model",
            on_click="ignore",
        )

        if test_df is not None:
            validate_feature_columns(test_df, features)
            validate_feature_schema(
                test_df,
                {feature: str(train_df.schema[feature]) for feature in features},
            )
            predictions = estimator.predict(test_df.select(features))
            results = prediction_frame(test_df, predictions, target)
            if target in test_df.columns:
                if not test_df.schema[target].is_numeric():
                    raise ValueError("The test target must be numeric to calculate metrics.")
                if test_df[target].null_count() > 0:
                    raise ValueError("The test target contains missing values.")
                _show_metrics(calculate_metrics(test_df[target], predictions))
            _show_predictions(results, key="download_training_predictions")
    except Exception as error:
        st.error(f"Training failed: {error}")


def _render_validation_mode() -> None:
    st.header("Validation")
    df = render_dataset_selector(
        "Upload training data",
        read_data,
        key="validation_data",
    )
    if df is None:
        st.info("Upload training data to validate a model.")
        return

    features, target = render_feature_target_selector(
        df,
        key_prefix="validation",
    )
    if not features:
        st.info("Select at least one feature to configure validation.")
        return
    strategy_column, validation_value_column = st.columns(2)
    strategy = strategy_column.selectbox(
        "Validation strategy",
        ["Random split", "Last split", "Cross-validation"],
        key="validation_strategy",
    )
    if strategy == "Cross-validation":
        folds = validation_value_column.slider(
            "Validation folds",
            2,
            10,
            5,
            key="validation_folds",
        )
        validation_percent = 20
    else:
        validation_percent = validation_value_column.slider(
            "Validation percent",
            1,
            50,
            20,
            key="validation_percent",
        )
        folds = 5

    try:
        preprocessing_df = _render_preprocessing(
            df,
            features,
            key_prefix="validation",
        )
    except ValueError as error:
        st.error(str(error))
        return

    model_config, model_config_is_valid = render_modeling_component(
        key_prefix="validation"
    )
    if not st.button(
        f"Validate {model_config.definition.label}",
        type="primary",
        disabled=not model_config_is_valid,
        key="validate_model",
    ):
        return

    try:
        _validate_training_columns(df, features, target)
        transformer = create_preprocessing_transformer(preprocessing_df)
        estimator = create_estimator(transformer, model_config)

        with st.spinner("Validating model..."):
            if strategy == "Cross-validation":
                metrics = cross_validate_estimator(
                    estimator,
                    df.select(features),
                    df[target],
                    folds,
                )
                _show_cross_validation_metrics(metrics)
            else:
                train_df, validation_df = split_validation_data(
                    df,
                    cast(ValidationStrategy, strategy),
                    validation_percent,
                )
                estimator.fit(train_df.select(features), train_df[target])
                predictions = estimator.predict(validation_df.select(features))
                _show_metrics(calculate_metrics(validation_df[target], predictions))
                _show_grid_results(estimator)
                _show_predictions(
                    prediction_frame(validation_df, predictions, target),
                    key="download_validation_predictions",
                )
    except Exception as error:
        st.error(f"Validation failed: {error}")


def _render_test_mode() -> None:
    st.header("Test")
    st.warning("Only upload model bundles that you created or trust.")
    model_column, test_data_column = st.columns(2)
    with model_column:
        model_file = st.file_uploader(
            "Upload model bundle",
            type=["joblib"],
            key="test_model",
        )
    with test_data_column:
        test_df = render_dataset_selector(
            "Upload test data",
            read_data,
            key="test_data",
        )
    if model_file is None or test_df is None:
        st.info("Upload a model bundle and test data to run predictions.")
        return

    try:
        artifact = deserialize_artifact(model_file.getvalue())
        st.caption(
            f"Model: {artifact.model_label} | Target: {artifact.target} | "
            f"Features: {', '.join(artifact.features)}"
        )
    except Exception as error:
        st.error(f"Could not load model bundle: {error}")
        return

    if not st.button("Run test predictions", type="primary", key="run_test"):
        return

    try:
        validate_feature_columns(test_df, artifact.features)
        validate_feature_schema(test_df, artifact.feature_dtypes)
        predictions = artifact.pipeline.predict(test_df.select(artifact.features))
        results = prediction_frame(test_df, predictions, artifact.target)
        if artifact.target in test_df.columns:
            if not test_df.schema[artifact.target].is_numeric():
                raise ValueError("The test target must be numeric to calculate metrics.")
            if test_df[artifact.target].null_count() > 0:
                raise ValueError("The test target contains missing values.")
            _show_metrics(calculate_metrics(test_df[artifact.target], predictions))
        _show_predictions(results, key="download_test_predictions")
    except Exception as error:
        st.error(f"Prediction failed: {error}")


st.set_page_config(page_title="MLStudio", layout="wide")
st.title("MLStudio")
st.caption("Train, validate, and test regression models.")

mode = st.radio(
    "Mode",
    ["Training", "Validation", "Test"],
    horizontal=True,
    key="mode",
)

if mode == "Training":
    _render_training_mode()
elif mode == "Validation":
    _render_validation_mode()
else:
    _render_test_mode()
