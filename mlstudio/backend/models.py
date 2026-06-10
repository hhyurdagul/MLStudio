from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)

ParameterValue = int | float | bool | str | None
ParameterKind = Literal["integer", "float", "boolean", "select"]


@dataclass(frozen=True)
class ModelParameter:
    name: str
    label: str
    kind: ParameterKind
    default: ParameterValue
    minimum: int | float | None = None
    maximum: int | float | None = None
    step: int | float | None = None
    options: tuple[ParameterValue, ...] = ()
    grid_options: tuple[ParameterValue, ...] = ()
    grid_default: tuple[ParameterValue, ...] = ()
    none_label: str | None = None


@dataclass(frozen=True)
class ModelDefinition:
    key: str
    label: str
    parameters: tuple[ModelParameter, ...]
    create_estimator: Callable[[dict[str, ParameterValue]], BaseEstimator]


def _create_random_forest(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return RandomForestRegressor(
        **parameters,
        random_state=42,
        n_jobs=-1,
    )


def _create_gradient_boosting(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return GradientBoostingRegressor(
        **parameters,
        random_state=42,
    )


def _create_voting_regressor(
    parameters: dict[str, ParameterValue],
) -> BaseEstimator:
    estimator = VotingRegressor(
        estimators=[
            (
                "random_forest",
                RandomForestRegressor(random_state=42, n_jobs=-1),
            ),
            (
                "gradient_boosting",
                GradientBoostingRegressor(random_state=42),
            ),
        ],
        n_jobs=-1,
    )
    return estimator.set_params(**parameters)


RANDOM_FOREST_PARAMETERS = (
    ModelParameter(
        name="n_estimators",
        label="Number of trees",
        kind="integer",
        default=100,
        minimum=10,
        maximum=2_000,
        step=10,
        grid_options=(10, 25, 50, 100, 200, 300, 500, 750, 1_000),
        grid_default=(100, 200, 500),
    ),
    ModelParameter(
        name="max_depth",
        label="Maximum depth",
        kind="select",
        default=None,
        options=(None, 2, 3, 5, 10, 15, 20, 30, 50),
        none_label="No limit",
        grid_options=(None, 2, 3, 5, 10, 15, 20, 30, 50),
        grid_default=(None, 5, 10, 20),
    ),
    ModelParameter(
        name="min_samples_split",
        label="Minimum samples split",
        kind="integer",
        default=2,
        minimum=2,
        maximum=100,
        step=1,
        grid_options=(2, 3, 4, 5, 8, 10, 15, 20),
        grid_default=(2, 5, 10),
    ),
)

GRADIENT_BOOSTING_PARAMETERS = (
    ModelParameter(
        name="n_estimators",
        label="Number of boosting stages",
        kind="integer",
        default=100,
        minimum=10,
        maximum=2_000,
        step=10,
        grid_options=(50, 100, 200, 300, 500),
        grid_default=(100, 200, 300),
    ),
    ModelParameter(
        name="learning_rate",
        label="Learning rate",
        kind="float",
        default=0.1,
        minimum=0.01,
        maximum=1.0,
        step=0.01,
        grid_options=(0.01, 0.05, 0.1, 0.2, 0.5),
        grid_default=(0.05, 0.1, 0.2),
    ),
    ModelParameter(
        name="max_depth",
        label="Maximum tree depth",
        kind="integer",
        default=3,
        minimum=1,
        maximum=30,
        step=1,
        grid_options=(1, 2, 3, 5, 8, 10),
        grid_default=(2, 3, 5),
    ),
)


def _prefixed_parameters(
    estimator_name: str,
    estimator_label: str,
    parameters: tuple[ModelParameter, ...],
) -> tuple[ModelParameter, ...]:
    return tuple(
        ModelParameter(
            name=f"{estimator_name}__{parameter.name}",
            label=f"{estimator_label}: {parameter.label}",
            kind=parameter.kind,
            default=parameter.default,
            minimum=parameter.minimum,
            maximum=parameter.maximum,
            step=parameter.step,
            options=parameter.options,
            grid_options=parameter.grid_options,
            grid_default=parameter.grid_default,
            none_label=parameter.none_label,
        )
        for parameter in parameters
    )


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    model.key: model
    for model in (
        ModelDefinition(
            key="random_forest",
            label="Random Forest Regressor",
            create_estimator=_create_random_forest,
            parameters=RANDOM_FOREST_PARAMETERS,
        ),
        ModelDefinition(
            key="gradient_boosting",
            label="Gradient Boosting Regressor",
            create_estimator=_create_gradient_boosting,
            parameters=GRADIENT_BOOSTING_PARAMETERS,
        ),
        ModelDefinition(
            key="voting_regressor",
            label="Voting Regressor",
            create_estimator=_create_voting_regressor,
            parameters=(
                *_prefixed_parameters(
                    "random_forest",
                    "Random Forest",
                    RANDOM_FOREST_PARAMETERS,
                ),
                *_prefixed_parameters(
                    "gradient_boosting",
                    "Gradient Boosting",
                    GRADIENT_BOOSTING_PARAMETERS,
                ),
            ),
        ),
    )
}


def get_model_definitions() -> dict[str, ModelDefinition]:
    return MODEL_REGISTRY


def format_parameter_value(parameter: ModelParameter) -> Callable[[Any], str]:
    def formatter(value: Any) -> str:
        if value is None and parameter.none_label is not None:
            return parameter.none_label
        return str(value)

    return formatter
