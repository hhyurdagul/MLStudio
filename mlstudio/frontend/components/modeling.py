from typing import Any

import streamlit as st

from mlstudio.backend import ModelConfig, get_model_definitions
from mlstudio.backend.models import (
    ModelDefinition,
    ModelParameter,
    ParameterValue,
    format_parameter_value,
    is_parameter_visible,
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
    if len(model.parameters) == 0:
        return {} 
    columns = st.columns(min(3, len(model.parameters)))
    parameters: dict[str, ParameterValue] = {}
    rendered_index = 0
    for parameter in model.parameters:
        if not is_parameter_visible(parameter, parameters):
            continue
        parameters[parameter.name] = _render_parameter(
            parameter,
            columns[rendered_index % len(columns)],
        )
        rendered_index += 1
    return parameters


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
    if len(model.parameters) == 0:
        return ({}), False
    st.caption("Select the values GridSearchCV should evaluate.")
    columns = st.columns(min(3, len(model.parameters)))
    param_grid: dict[str, list[ParameterValue]] = {}
    valid = True
    selected_values: dict[str, list[ParameterValue]] = {}
    rendered_index = 0
    for parameter in model.parameters:
        if not is_parameter_visible(parameter, selected_values):
            continue
        values = columns[rendered_index % len(columns)].multiselect(
            parameter.label,
            options=parameter.grid_options,
            default=parameter.grid_default,
            format_func=format_parameter_value(parameter),
        )
        selected_values[parameter.name] = values
        param_grid[f"model__{parameter.name}"] = values
        valid = valid and bool(values)
        rendered_index += 1
    if not valid:
        st.error("Every grid-search parameter needs at least one value.")
    return param_grid, valid
