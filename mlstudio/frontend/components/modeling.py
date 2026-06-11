from typing import Any

import streamlit as st

from mlstudio.backend import ModelConfig, get_model_definitions
from mlstudio.backend.models import (
    ModelDefinition,
    ModelParameter,
    ParameterValue,
    format_parameter_value,
)


def render_model_config() -> tuple[ModelConfig, bool]:
    models = get_model_definitions()
    model_key = st.selectbox(
        "Model",
        options=list(models),
        format_func=lambda key: models[key].label,
    )
    model = models[model_key]
    parameters = _render_parameters(model)
    use_grid_search = st.checkbox(
        "Use grid search",
    )
    cv = (
        st.slider(
            "Grid-search folds",
            2,
            10,
            5,
        )
        if use_grid_search
        else 5
    )
    param_grid, grid_is_valid = _render_grid(model) if use_grid_search else ({}, True)
    return (
        ModelConfig(model, parameters, use_grid_search, param_grid, cv),
        grid_is_valid,
    )


def _render_parameters(
    model: ModelDefinition,
) -> dict[str, ParameterValue]:
    columns = st.columns(min(3, len(model.parameters)))
    return {
        parameter.name: _render_parameter(
            parameter,
            columns[index % len(columns)],
        )
        for index, parameter in enumerate(model.parameters)
    }


def _render_parameter(
    parameter: ModelParameter,
    container: Any,
) -> ParameterValue:
    if parameter.kind == "integer" or parameter.kind == "float":
        cast = int if parameter.kind == "integer" else float
        return cast(
            container.number_input(
                parameter.label,
                value=parameter.default,
                min_value=parameter.minimum,
                max_value=parameter.maximum,
                step=parameter.step,
            )
        )
    if parameter.kind == "boolean":
        return bool(
            container.checkbox(
                parameter.label,
                value=parameter.default,
            )
        )
    return container.selectbox(
        parameter.label,
        options=parameter.options,
        index=parameter.options.index(parameter.default),
        format_func=format_parameter_value(parameter),
    )


def _render_grid(
    model: ModelDefinition,
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
        )
        param_grid[f"model__{parameter.name}"] = values
        valid = valid and bool(values)
    if not valid:
        st.error("Every grid-search parameter needs at least one value.")
    return param_grid, valid
