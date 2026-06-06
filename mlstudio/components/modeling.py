from typing import Any, cast

import polars as pl
import streamlit as st
from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline

from mlstudio.ml.models import (
    ModelDefinition,
    ModelParameter,
    ParameterValue,
    format_parameter_value,
    get_model_definitions,
)


def _render_parameter(
    model: ModelDefinition,
    parameter: ModelParameter,
    container: Any,
) -> ParameterValue:
    key = f"model_{model.key}_{parameter.name}"
    common = {
        "label": parameter.label,
        "key": key,
    }

    if parameter.kind == "integer":
        return int(
            container.number_input(
                **common,
                value=cast(int, parameter.default),
                min_value=cast(int, parameter.minimum),
                max_value=cast(int, parameter.maximum),
                step=cast(int, parameter.step),
            )
        )
    if parameter.kind == "float":
        return float(
            container.number_input(
                **common,
                value=cast(float, parameter.default),
                min_value=cast(float, parameter.minimum),
                max_value=cast(float, parameter.maximum),
                step=cast(float, parameter.step),
            )
        )
    if parameter.kind == "boolean":
        return bool(container.checkbox(**common, value=cast(bool, parameter.default)))

    return container.selectbox(
        **common,
        options=parameter.options,
        index=parameter.options.index(parameter.default),
        format_func=format_parameter_value(parameter),
    )


def _render_model_parameters(
    model: ModelDefinition,
) -> dict[str, ParameterValue]:
    columns = st.columns(min(3, len(model.parameters)))
    return {
        parameter.name: _render_parameter(
            model,
            parameter,
            columns[index % len(columns)],
        )
        for index, parameter in enumerate(model.parameters)
    }


def _render_grid_parameters(
    model: ModelDefinition,
) -> tuple[dict[str, list[ParameterValue]], bool]:
    st.write("Grid Search Parameters")
    st.caption("Select the values that GridSearchCV should evaluate.")

    param_grid: dict[str, list[ParameterValue]] = {}
    empty_parameters: list[str] = []
    columns = st.columns(min(3, len(model.parameters)))

    for index, parameter in enumerate(model.parameters):
        values = columns[index % len(columns)].multiselect(
            parameter.label,
            options=parameter.grid_options,
            default=parameter.grid_default,
            format_func=format_parameter_value(parameter),
            key=f"grid_{model.key}_{parameter.name}",
        )
        param_grid[f"model__{parameter.name}"] = values
        if not values:
            empty_parameters.append(parameter.label)

    if empty_parameters:
        st.error("Select at least one value for: " + ", ".join(empty_parameters) + ".")

    return param_grid, not empty_parameters


def _create_pipeline(
    transformer: ColumnTransformer,
    estimator: BaseEstimator,
    use_grid_search: bool,
    cv: int,
    param_grid: dict[str, list[ParameterValue]],
) -> Pipeline | GridSearchCV:
    pipeline = Pipeline(
        [
            ("preprocessing", transformer),
            ("model", estimator),
        ]
    )
    if not use_grid_search:
        return pipeline

    return GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        n_jobs=-1,
        return_train_score=True,
    )


def render_modeling_component(
    df: pl.DataFrame,
    features: list[str],
    target: str,
    transformer: ColumnTransformer,
) -> None:
    models = get_model_definitions()
    model_by_label = {model.label: model for model in models}
    selected_label = st.selectbox("Model", model_by_label)
    model = model_by_label[selected_label]

    target_is_valid = model.target_type != "numeric" or df[target].dtype.is_numeric()
    if not target_is_valid:
        st.error(f"{model.label} requires a numeric target.")

    parameters = _render_model_parameters(model)

    test_column, grid_column = st.columns(2)
    test_size = test_column.slider("Test size", 0.1, 0.5, 0.2, 0.05)
    use_grid_search = grid_column.checkbox("Use grid search")
    cv = (
        grid_column.slider("Cross-validation folds", 2, 10, 5) if use_grid_search else 5
    )

    param_grid: dict[str, list[ParameterValue]] = {}
    grid_search_is_valid = True
    if use_grid_search:
        param_grid, grid_search_is_valid = _render_grid_parameters(model)

    if st.button(f"Train {model.label}", type="primary"):
        if not features:
            st.error("Select at least one feature.")
            return
        if not target_is_valid or not grid_search_is_valid:
            return

        estimator = _create_pipeline(
            transformer,
            model.create_estimator(parameters),
            use_grid_search,
            cv,
            param_grid,
        )
        x_train, x_test, y_train, y_test = train_test_split(
            df.select(features),
            df[target],
            test_size=test_size,
            random_state=42,
        )

        with st.spinner(f"Training {model.label}..."):
            estimator.fit(x_train, y_train)
            predictions = estimator.predict(x_test)

        st.metric(model.metric_name, f"{r2_score(y_test, predictions):.4f}")
        if isinstance(estimator, GridSearchCV):
            st.write("Best Parameters")
            st.json(estimator.best_params_)
            st.metric(
                "Best Cross-validation Score",
                f"{estimator.best_score_:.4f}",
            )
