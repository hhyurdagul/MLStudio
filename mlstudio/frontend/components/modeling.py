from typing import Any, cast

import streamlit as st

from mlstudio.backend import ModelConfig, get_model_definitions
from mlstudio.backend.models import (
    ModelDefinition,
    ModelParameter,
    ParameterValue,
    format_parameter_value,
)


def render_model_config(*, key_prefix: str) -> tuple[ModelConfig, bool]:
    models = get_model_definitions()
    model_key = st.selectbox(
        "Model",
        options=list(models),
        format_func=lambda key: models[key].label,
        key=f"{key_prefix}_model",
    )
    model = models[model_key]
    parameters = _render_parameters(model, key_prefix)
    use_grid_search = st.checkbox(
        "Use grid search",
        key=f"{key_prefix}_use_grid_search",
    )
    cv = (
        st.slider(
            "Grid-search folds",
            2,
            10,
            5,
            key=f"{key_prefix}_grid_cv",
        )
        if use_grid_search
        else 5
    )
    param_grid, grid_is_valid = (
        _render_grid(model, key_prefix) if use_grid_search else ({}, True)
    )
    return (
        ModelConfig(model, parameters, use_grid_search, param_grid, cv),
        grid_is_valid,
    )


def _render_parameters(
    model: ModelDefinition,
    key_prefix: str,
) -> dict[str, ParameterValue]:
    columns = st.columns(min(3, len(model.parameters)))
    return {
        parameter.name: _render_parameter(
            model,
            parameter,
            columns[index % len(columns)],
            key_prefix,
        )
        for index, parameter in enumerate(model.parameters)
    }


def _render_parameter(
    model: ModelDefinition,
    parameter: ModelParameter,
    container: Any,
    key_prefix: str,
) -> ParameterValue:
    key = f"{key_prefix}_{model.key}_{parameter.name}"
    if parameter.kind == "integer":
        return int(
            container.number_input(
                parameter.label,
                value=cast(int, parameter.default),
                min_value=cast(int, parameter.minimum),
                max_value=cast(int, parameter.maximum),
                step=cast(int, parameter.step),
                key=key,
            )
        )
    if parameter.kind == "float":
        return float(
            container.number_input(
                parameter.label,
                value=cast(float, parameter.default),
                min_value=cast(float, parameter.minimum),
                max_value=cast(float, parameter.maximum),
                step=cast(float, parameter.step),
                key=key,
            )
        )
    if parameter.kind == "boolean":
        return bool(
            container.checkbox(
                parameter.label,
                value=cast(bool, parameter.default),
                key=key,
            )
        )
    return container.selectbox(
        parameter.label,
        options=parameter.options,
        index=parameter.options.index(parameter.default),
        format_func=format_parameter_value(parameter),
        key=key,
    )


def _render_grid(
    model: ModelDefinition,
    key_prefix: str,
) -> tuple[dict[str, list[ParameterValue]], bool]:
    st.caption("Select the values GridSearchCV should evaluate.")
    columns = st.columns(min(3, len(model.parameters)))
    param_grid: dict[str, list[ParameterValue]] = {}
    valid = True
    for index, parameter in enumerate(model.parameters):
        values = columns[index % len(columns)].multiselect(
            parameter.label,
            options=parameter.grid_options,
            default=parameter.grid_default,
            format_func=format_parameter_value(parameter),
            key=f"{key_prefix}_grid_{model.key}_{parameter.name}",
        )
        param_grid[f"model__{parameter.name}"] = values
        valid = valid and bool(values)
    if not valid:
        st.error("Every grid-search parameter needs at least one value.")
    return param_grid, valid
