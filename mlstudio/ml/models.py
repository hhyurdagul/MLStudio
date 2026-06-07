from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

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


def _create_random_forest(
    parameters: dict[str, ParameterValue],
) -> BaseEstimator:
    return RandomForestRegressor(
        **parameters,
        random_state=42,
        n_jobs=-1,
    )


def _create_gradient_boosting(
    parameters: dict[str, ParameterValue],
) -> BaseEstimator:
    return GradientBoostingRegressor(
        **parameters,
        random_state=42,
    )


def _create_ridge(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return Ridge(**parameters)


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    model.key: model
    for model in (
        ModelDefinition(
            key="random_forest",
            label="Random Forest Regressor",
            create_estimator=_create_random_forest,
            parameters=(
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
            ),
        ),
        ModelDefinition(
            key="gradient_boosting",
            label="Gradient Boosting Regressor",
            create_estimator=_create_gradient_boosting,
            parameters=(
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
            ),
        ),
        ModelDefinition(
            key="ridge",
            label="Ridge Regression",
            create_estimator=_create_ridge,
            parameters=(
                ModelParameter(
                    name="alpha",
                    label="Regularization strength",
                    kind="float",
                    default=1.0,
                    minimum=0.0,
                    maximum=100.0,
                    step=0.1,
                    grid_options=(0.0, 0.01, 0.1, 1.0, 10.0, 100.0),
                    grid_default=(0.1, 1.0, 10.0),
                ),
                ModelParameter(
                    name="fit_intercept",
                    label="Fit intercept",
                    kind="boolean",
                    default=True,
                    grid_options=(True, False),
                    grid_default=(True, False),
                ),
            ),
        ),
    )
}


def get_model_definitions() -> dict[str, ModelDefinition]:
    return MODEL_REGISTRY


def format_parameter_value(
    parameter: ModelParameter,
) -> Callable[[Any], str]:
    def formatter(value: Any) -> str:
        if value is None and parameter.none_label is not None:
            return parameter.none_label
        return str(value)

    return formatter
