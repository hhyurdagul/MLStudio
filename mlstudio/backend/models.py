from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    GradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .elm import ELMActivation, ELMRegressor

ParameterValue = int | float | bool | str | None
ParameterKind = Literal["integer", "float", "boolean", "select"]


@dataclass(frozen=True)
class ParameterCondition:
    parameter: str
    values: tuple[ParameterValue, ...]


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
    visible_when: tuple[ParameterCondition, ...] = ()


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


def _create_svr(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return SVR(**parameters)


def _create_elm(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return ELMRegressor(
        n_hidden=int(cast(int | float, parameters["n_hidden"])),
        activation=cast(ELMActivation, parameters["activation"]),
        alpha=float(cast(int | float, parameters["alpha"])),
        random_state=42,
    )


def _create_xgboost(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return XGBRegressor(
        **parameters,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )


def _create_catboost(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return cast(
        BaseEstimator,
        CatBoostRegressor(
            **parameters,
            random_seed=42,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        ),
    )


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
        label="Number of trees",
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

SVM_PARAMETERS = (
    ModelParameter(
        name="kernel",
        label="Kernel",
        kind="select",
        default="rbf",
        options=("linear", "poly", "rbf", "sigmoid"),
        grid_options=("linear", "poly", "rbf", "sigmoid"),
        grid_default=("rbf",),
    ),
    ModelParameter(
        name="C",
        label="Regularization (C)",
        kind="float",
        default=1.0,
        minimum=0.001,
        maximum=1_000.0,
        step=0.1,
        grid_options=(0.1, 1.0, 10.0, 100.0),
        grid_default=(0.1, 1.0, 10.0),
    ),
    ModelParameter(
        name="epsilon",
        label="Epsilon",
        kind="float",
        default=0.1,
        minimum=0.0,
        maximum=10.0,
        step=0.01,
        grid_options=(0.01, 0.05, 0.1, 0.2, 0.5),
        grid_default=(0.05, 0.1, 0.2),
    ),
    ModelParameter(
        name="degree",
        label="Polynomial degree",
        kind="integer",
        default=3,
        minimum=1,
        maximum=10,
        step=1,
        grid_options=(2, 3, 4, 5),
        grid_default=(2, 3, 4),
        visible_when=(ParameterCondition("kernel", ("poly",)),),
    ),
    ModelParameter(
        name="gamma",
        label="Kernel coefficient (gamma)",
        kind="select",
        default="scale",
        options=("scale", "auto"),
        grid_options=("scale", "auto"),
        grid_default=("scale", "auto"),
        visible_when=(ParameterCondition("kernel", ("poly", "rbf", "sigmoid")),),
    ),
    ModelParameter(
        name="coef0",
        label="Independent kernel term (coef)",
        kind="float",
        default=0.0,
        minimum=-10.0,
        maximum=10.0,
        step=0.1,
        grid_options=(-1.0, 0.0, 0.5, 1.0),
        grid_default=(0.0, 0.5, 1.0),
        visible_when=(ParameterCondition("kernel", ("poly", "sigmoid")),),
    ),
)

ELM_PARAMETERS = (
    ModelParameter(
        name="n_hidden",
        label="Hidden neurons",
        kind="integer",
        default=100,
        minimum=1,
        maximum=5_000,
        step=10,
        grid_options=(25, 50, 100, 200, 500, 1_000),
        grid_default=(50, 100, 200),
    ),
    ModelParameter(
        name="activation",
        label="Activation",
        kind="select",
        default="tanh",
        options=("relu", "tanh", "sigmoid", "sine"),
        grid_options=("relu", "tanh", "sigmoid", "sine"),
        grid_default=("relu", "tanh", "sigmoid"),
    ),
    ModelParameter(
        name="alpha",
        label="Ridge regularization",
        kind="float",
        default=0.001,
        minimum=0.0,
        maximum=100.0,
        step=0.001,
        grid_options=(0.0, 0.0001, 0.001, 0.01, 0.1, 1.0),
        grid_default=(0.0001, 0.001, 0.01),
    ),
)

XGBOOST_PARAMETERS = (
    ModelParameter(
        name="n_estimators",
        label="Number of trees",
        kind="integer",
        default=100,
        minimum=10,
        maximum=5_000,
        step=10,
        grid_options=(50, 100, 200, 300, 500, 1_000),
        grid_default=(100, 200, 500),
    ),
    ModelParameter(
        name="max_depth",
        label="Maximum tree depth",
        kind="integer",
        default=6,
        minimum=1,
        maximum=30,
        step=1,
        grid_options=(2, 3, 5, 6, 8, 10),
        grid_default=(3, 6, 10),
    ),
    ModelParameter(
        name="learning_rate",
        label="Learning rate",
        kind="float",
        default=0.3,
        minimum=0.001,
        maximum=1.0,
        step=0.01,
        grid_options=(0.01, 0.05, 0.1, 0.2, 0.3),
        grid_default=(0.05, 0.1, 0.3),
    ),
)

CATBOOST_PARAMETERS = (
    ModelParameter(
        name="iterations",
        label="Number of boosting iterations",
        kind="integer",
        default=500,
        minimum=10,
        maximum=5_000,
        step=10,
        grid_options=(100, 300, 500, 750, 1_000),
        grid_default=(300, 500, 1_000),
    ),
    ModelParameter(
        name="depth",
        label="Tree depth",
        kind="integer",
        default=6,
        minimum=1,
        maximum=16,
        step=1,
        grid_options=(4, 6, 8, 10),
        grid_default=(4, 6, 8),
    ),
    ModelParameter(
        name="learning_rate",
        label="Learning rate",
        kind="float",
        default=0.03,
        minimum=0.001,
        maximum=1.0,
        step=0.01,
        grid_options=(0.01, 0.03, 0.05, 0.1, 0.2),
        grid_default=(0.03, 0.05, 0.1),
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
            visible_when=tuple(
                ParameterCondition(
                    parameter=f"{estimator_name}__{condition.parameter}",
                    values=condition.values,
                )
                for condition in parameter.visible_when
            ),
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
            key="svm",
            label="Support Vector Regressor",
            create_estimator=_create_svr,
            parameters=SVM_PARAMETERS,
        ),
        ModelDefinition(
            key="elm",
            label="Extreme Learning Machine Regressor",
            create_estimator=_create_elm,
            parameters=ELM_PARAMETERS,
        ),
        ModelDefinition(
            key="xgboost",
            label="XGBoost Regressor",
            create_estimator=_create_xgboost,
            parameters=XGBOOST_PARAMETERS,
        ),
        ModelDefinition(
            key="catboost",
            label="CatBoost Regressor",
            create_estimator=_create_catboost,
            parameters=CATBOOST_PARAMETERS,
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


def is_parameter_visible(
    parameter: ModelParameter,
    selected_values: dict[str, Any],
) -> bool:
    for condition in parameter.visible_when:
        selected = selected_values.get(condition.parameter)
        if isinstance(selected, list | tuple | set):
            if not any(value in condition.values for value in selected):
                return False
        elif selected not in condition.values:
            return False
    return True


def format_parameter_value(parameter: ModelParameter) -> Callable[[Any], str]:
    def formatter(value: Any) -> str:
        if value is None and parameter.none_label is not None:
            return parameter.none_label
        return str(value)

    return formatter
