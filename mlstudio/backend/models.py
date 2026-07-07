from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal, cast

from catboost import CatBoostRegressor
from sklearn.base import BaseEstimator
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import (
    AdaBoostRegressor,
    BaggingRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
    VotingRegressor,
)
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import (
    BayesianRidge,
    ElasticNet,
    HuberRegressor,
    Lars,
    Lasso,
    LassoLars,
    LinearRegression,
    QuantileRegressor,
    Ridge,
    SGDRegressor,
    TweedieRegressor,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.svm import NuSVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from lightgbm import LGBMRegressor
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


@dataclass(frozen=True)
class VotingEstimatorDefinition:
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


def _create_linear_regression(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return LinearRegression(**parameters)


def _create_ridge(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return Ridge(**parameters, random_state=42)


def _create_lasso(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return Lasso(**parameters, random_state=42)


def _create_elastic_net(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return ElasticNet(**parameters, random_state=42)


def _create_lars(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return Lars(**parameters)


def _create_lasso_lars(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return LassoLars(**parameters)


def _create_bayesian_ridge(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return BayesianRidge(**parameters)

def _create_huber_regressor(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return HuberRegressor(**parameters)
def _create_sgd_regressor(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return SGDRegressor(**parameters, random_state=42)

def _create_quantile_regressor(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return QuantileRegressor(**parameters)


def _create_tweedie_regressor(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return TweedieRegressor(**parameters)


def _create_decision_tree(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return DecisionTreeRegressor(**parameters, random_state=42)


def _create_extra_tree(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return ExtraTreeRegressor(**parameters, random_state=42)


def _create_extra_trees(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return ExtraTreesRegressor(**parameters, random_state=42, n_jobs=-1)


def _create_adaboost(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return AdaBoostRegressor(**parameters, random_state=42)


def _create_bagging(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return BaggingRegressor(**parameters, random_state=42, n_jobs=-1)


def _create_hist_gradient_boosting(
    parameters: dict[str, ParameterValue],
) -> BaseEstimator:
    return HistGradientBoostingRegressor(**parameters, random_state=42)


def _create_knn(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return KNeighborsRegressor(**parameters, n_jobs=-1)


def _default_parameters(
    parameters: tuple[ModelParameter, ...],
) -> dict[str, ParameterValue]:
    return {parameter.name: parameter.default for parameter in parameters}


def _create_voting_regressor(
    parameters: dict[str, ParameterValue],
) -> BaseEstimator:
    estimator = VotingRegressor(
        estimators=[
            (
                model.key,
                model.create_estimator(_default_parameters(model.parameters)),
            )
            for model in VOTING_REGRESSOR_MODELS
        ],
        n_jobs=-1,
    )
    return estimator.set_params(**parameters)


def _create_svr(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return SVR(**parameters)


def _create_nu_svr(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return NuSVR(**parameters)


def _create_kernel_ridge(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return KernelRidge(**parameters)


def _create_gaussian_process(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return GaussianProcessRegressor(**parameters, random_state=42)


def _create_mlp(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    hidden_layers = tuple(
        int(value.strip())
        for value in str(parameters["hidden_layer_sizes"]).split(",")
        if value.strip()
    )
    return MLPRegressor(
        **{**parameters, "hidden_layer_sizes": hidden_layers},
        random_state=42,
    )


def _create_pls_regression(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return PLSRegression(**parameters)



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


def _create_lightgbm(parameters: dict[str, ParameterValue]) -> BaseEstimator:
    return LGBMRegressor(
        **parameters,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
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

LINEAR_REGRESSION_PARAMETERS: tuple[ModelParameter, ...] = ()

REGULARIZED_LINEAR_PARAMETERS = (
    ModelParameter(
        name="alpha",
        label="Regularization strength",
        kind="float",
        default=1.0,
        minimum=0.0,
        maximum=1_000.0,
        step=0.01,
        grid_options=(0.0001, 0.001, 0.01, 0.1, 1.0, 10.0),
        grid_default=(0.01, 0.1, 1.0),
    ),
)

ELASTIC_NET_PARAMETERS = (
    *REGULARIZED_LINEAR_PARAMETERS,
    ModelParameter(
        name="l1_ratio",
        label="L1 ratio",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        grid_options=(0.1, 0.25, 0.5, 0.75, 0.9),
        grid_default=(0.25, 0.5, 0.75),
    ),
)

LARS_PARAMETERS = (
    ModelParameter(
        name="n_nonzero_coefs",
        label="Non-zero coefficients",
        kind="select",
        default=500,
        options=(1, 5, 10, 25, 50, 100, 500),
        grid_options=(1, 5, 10, 25, 50, 100, 500),
        grid_default=(10, 50, 100),
    ),
)

LASSO_LARS_PARAMETERS = REGULARIZED_LINEAR_PARAMETERS

ORTHOGONAL_MATCHING_PURSUIT_PARAMETERS = (
    ModelParameter(
        name="n_nonzero_coefs",
        label="Non-zero coefficients",
        kind="select",
        default=None,
        options=(None, 1, 2, 5, 10, 25, 50, 100),
        none_label="Auto",
        grid_options=(None, 1, 2, 5, 10, 25, 50),
        grid_default=(None, 5, 10),
    ),
)

BAYESIAN_RIDGE_PARAMETERS = (
    ModelParameter(
        name="max_iter",
        label="Maximum iterations",
        kind="integer",
        default=300,
        minimum=10,
        maximum=5_000,
        step=10,
        grid_options=(100, 300, 500, 1_000),
        grid_default=(300, 500),
    ),
)

ARD_REGRESSION_PARAMETERS = BAYESIAN_RIDGE_PARAMETERS

HUBER_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="epsilon",
        label="Outlier threshold",
        kind="float",
        default=1.35,
        minimum=1.0,
        maximum=10.0,
        step=0.05,
        grid_options=(1.1, 1.35, 1.5, 2.0),
        grid_default=(1.35, 1.5, 2.0),
    ),
    ModelParameter(
        name="alpha",
        label="Regularization strength",
        kind="float",
        default=0.0001,
        minimum=0.0,
        maximum=10.0,
        step=0.0001,
        grid_options=(0.0, 0.0001, 0.001, 0.01, 0.1),
        grid_default=(0.0001, 0.001, 0.01),
    ),
)

RANSAC_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="min_samples",
        label="Minimum sample fraction",
        kind="float",
        default=None,
        minimum=0.0,
        maximum=1.0,
        step=0.05,
        grid_options=(None, 0.25, 0.5, 0.75),
        grid_default=(None, 0.5),
        none_label="Auto",
    ),
)

THEIL_SEN_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="max_subpopulation",
        label="Maximum subpopulation",
        kind="integer",
        default=10_000,
        minimum=100,
        maximum=100_000,
        step=100,
        grid_options=(1_000, 5_000, 10_000, 25_000),
        grid_default=(5_000, 10_000),
    ),
)

SGD_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="loss",
        label="Loss",
        kind="select",
        default="squared_error",
        options=("squared_error", "huber", "epsilon_insensitive"),
        grid_options=("squared_error", "huber", "epsilon_insensitive"),
        grid_default=("squared_error", "huber"),
    ),
    ModelParameter(
        name="alpha",
        label="Regularization strength",
        kind="float",
        default=0.0001,
        minimum=0.0,
        maximum=10.0,
        step=0.0001,
        grid_options=(0.0001, 0.001, 0.01, 0.1),
        grid_default=(0.0001, 0.001, 0.01),
    ),
    ModelParameter(
        name="penalty",
        label="Penalty",
        kind="select",
        default="l2",
        options=("l2", "l1", "elasticnet"),
        grid_options=("l2", "l1", "elasticnet"),
        grid_default=("l2", "l1"),
    ),
)

PASSIVE_AGGRESSIVE_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="C",
        label="Regularization (C)",
        kind="float",
        default=1.0,
        minimum=0.001,
        maximum=1_000.0,
        step=0.1,
        grid_options=(0.1, 1.0, 10.0),
        grid_default=(0.1, 1.0, 10.0),
    ),
    ModelParameter(
        name="loss",
        label="Loss",
        kind="select",
        default="epsilon_insensitive",
        options=("epsilon_insensitive", "squared_epsilon_insensitive"),
        grid_options=("epsilon_insensitive", "squared_epsilon_insensitive"),
        grid_default=("epsilon_insensitive",),
    ),
)

QUANTILE_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="quantile",
        label="Quantile",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        grid_options=(0.1, 0.25, 0.5, 0.75, 0.9),
        grid_default=(0.25, 0.5, 0.75),
    ),
    *REGULARIZED_LINEAR_PARAMETERS,
)

TWEEDIE_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="power",
        label="Distribution power",
        kind="float",
        default=0.0,
        minimum=0.0,
        maximum=3.0,
        step=0.1,
        grid_options=(0.0, 1.0, 1.5, 2.0),
        grid_default=(0.0, 1.0, 2.0),
    ),
    *REGULARIZED_LINEAR_PARAMETERS,
)

TREE_PARAMETERS = (
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

EXTRA_TREES_PARAMETERS = (
    *RANDOM_FOREST_PARAMETERS,
)

ADABOOST_PARAMETERS = (
    ModelParameter(
        name="n_estimators",
        label="Number of estimators",
        kind="integer",
        default=50,
        minimum=10,
        maximum=2_000,
        step=10,
        grid_options=(50, 100, 200, 300, 500),
        grid_default=(50, 100, 200),
    ),
    ModelParameter(
        name="learning_rate",
        label="Learning rate",
        kind="float",
        default=1.0,
        minimum=0.001,
        maximum=10.0,
        step=0.01,
        grid_options=(0.01, 0.1, 0.5, 1.0),
        grid_default=(0.1, 0.5, 1.0),
    ),
    ModelParameter(
        name="loss",
        label="Loss",
        kind="select",
        default="linear",
        options=("linear", "square", "exponential"),
        grid_options=("linear", "square", "exponential"),
        grid_default=("linear", "square"),
    ),
)

BAGGING_PARAMETERS = (
    ModelParameter(
        name="n_estimators",
        label="Number of estimators",
        kind="integer",
        default=10,
        minimum=1,
        maximum=1_000,
        step=1,
        grid_options=(10, 25, 50, 100, 200),
        grid_default=(10, 50, 100),
    ),
    ModelParameter(
        name="max_samples",
        label="Maximum sample fraction",
        kind="float",
        default=1.0,
        minimum=0.1,
        maximum=1.0,
        step=0.05,
        grid_options=(0.5, 0.75, 1.0),
        grid_default=(0.5, 1.0),
    ),
)

HIST_GRADIENT_BOOSTING_PARAMETERS = (
    ModelParameter(
        name="max_iter",
        label="Maximum iterations",
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
        minimum=0.001,
        maximum=1.0,
        step=0.01,
        grid_options=(0.01, 0.05, 0.1, 0.2),
        grid_default=(0.05, 0.1, 0.2),
    ),
    ModelParameter(
        name="max_leaf_nodes",
        label="Maximum leaf nodes",
        kind="select",
        default=31,
        options=(None, 15, 31, 63, 127),
        none_label="No limit",
        grid_options=(15, 31, 63, 127),
        grid_default=(31, 63),
    ),
)

KNN_PARAMETERS = (
    ModelParameter(
        name="n_neighbors",
        label="Neighbors",
        kind="integer",
        default=5,
        minimum=1,
        maximum=200,
        step=1,
        grid_options=(1, 3, 5, 7, 10, 15, 25),
        grid_default=(3, 5, 10),
    ),
    ModelParameter(
        name="weights",
        label="Weights",
        kind="select",
        default="uniform",
        options=("uniform", "distance"),
        grid_options=("uniform", "distance"),
        grid_default=("uniform", "distance"),
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

LINEAR_SVR_PARAMETERS = (
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
        default=0.0,
        minimum=0.0,
        maximum=10.0,
        step=0.01,
        grid_options=(0.0, 0.05, 0.1, 0.2),
        grid_default=(0.0, 0.1),
    ),
    ModelParameter(
        name="loss",
        label="Loss",
        kind="select",
        default="epsilon_insensitive",
        options=("epsilon_insensitive", "squared_epsilon_insensitive"),
        grid_options=("epsilon_insensitive", "squared_epsilon_insensitive"),
        grid_default=("epsilon_insensitive",),
    ),
)

NU_SVR_PARAMETERS = (
    ModelParameter(
        name="nu",
        label="Nu",
        kind="float",
        default=0.5,
        minimum=0.01,
        maximum=1.0,
        step=0.01,
        grid_options=(0.25, 0.5, 0.75),
        grid_default=(0.25, 0.5, 0.75),
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
        name="kernel",
        label="Kernel",
        kind="select",
        default="rbf",
        options=("linear", "poly", "rbf", "sigmoid"),
        grid_options=("linear", "poly", "rbf", "sigmoid"),
        grid_default=("rbf",),
    ),
)

KERNEL_RIDGE_PARAMETERS = (
    *REGULARIZED_LINEAR_PARAMETERS,
    ModelParameter(
        name="kernel",
        label="Kernel",
        kind="select",
        default="linear",
        options=("linear", "poly", "rbf", "sigmoid"),
        grid_options=("linear", "poly", "rbf", "sigmoid"),
        grid_default=("linear", "rbf"),
    ),
)

GAUSSIAN_PROCESS_PARAMETERS = (
    ModelParameter(
        name="alpha",
        label="Noise level",
        kind="float",
        default=0.0000000001,
        minimum=0.0,
        maximum=1.0,
        step=0.0001,
        grid_options=(0.0000000001, 0.000001, 0.0001, 0.01),
        grid_default=(0.0000000001, 0.000001, 0.0001),
    ),
)

MLP_PARAMETERS = (
    ModelParameter(
        name="hidden_layer_sizes",
        label="Hidden layer sizes",
        kind="select",
        default="100",
        options=("25", "50", "100", "100,50", "100,100"),
        grid_options=("25", "50", "100", "100,50", "100,100"),
        grid_default=("50", "100", "100,50"),
    ),
    ModelParameter(
        name="activation",
        label="Activation",
        kind="select",
        default="relu",
        options=("identity", "logistic", "tanh", "relu"),
        grid_options=("tanh", "relu"),
        grid_default=("relu",),
    ),
    ModelParameter(
        name="alpha",
        label="Regularization strength",
        kind="float",
        default=0.0001,
        minimum=0.0,
        maximum=10.0,
        step=0.0001,
        grid_options=(0.0001, 0.001, 0.01, 0.1),
        grid_default=(0.0001, 0.001, 0.01),
    ),
    ModelParameter(
        name="max_iter",
        label="Maximum iterations",
        kind="integer",
        default=200,
        minimum=50,
        maximum=5_000,
        step=50,
        grid_options=(200, 500, 1_000),
        grid_default=(200, 500),
    ),
)

PLS_REGRESSION_PARAMETERS = (
    ModelParameter(
        name="n_components",
        label="Components",
        kind="integer",
        default=1,
        minimum=1,
        maximum=100,
        step=1,
        grid_options=(1, 2, 3, 5, 10),
        grid_default=(1, 2, 3),
    ),
)

DUMMY_REGRESSOR_PARAMETERS = (
    ModelParameter(
        name="strategy",
        label="Strategy",
        kind="select",
        default="mean",
        options=("mean", "median", "quantile"),
        grid_options=("mean", "median", "quantile"),
        grid_default=("mean", "median"),
    ),
    ModelParameter(
        name="quantile",
        label="Quantile",
        kind="float",
        default=0.5,
        minimum=0.0,
        maximum=1.0,
        step=0.01,
        grid_options=(0.25, 0.5, 0.75),
        grid_default=(0.5,),
        visible_when=(ParameterCondition("strategy", ("quantile",)),),
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

LIGHTGBM_PARAMETERS = (
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
        name="learning_rate",
        label="Learning rate",
        kind="float",
        default=0.1,
        minimum=0.001,
        maximum=1.0,
        step=0.01,
        grid_options=(0.01, 0.05, 0.1, 0.2),
        grid_default=(0.05, 0.1, 0.2),
    ),
    ModelParameter(
        name="max_depth",
        label="Maximum tree depth",
        kind="integer",
        default=-1,
        minimum=-1,
        maximum=30,
        step=1,
        grid_options=(-1, 3, 5, 7, 10),
        grid_default=(-1, 5, 10),
    ),
    ModelParameter(
        name="num_leaves",
        label="Number of leaves",
        kind="integer",
        default=31,
        minimum=2,
        maximum=1_024,
        step=1,
        grid_options=(15, 31, 63, 127),
        grid_default=(31, 63),
    ),
    ModelParameter(
        name="subsample",
        label="Subsample fraction",
        kind="float",
        default=1.0,
        minimum=0.1,
        maximum=1.0,
        step=0.05,
        grid_options=(0.6, 0.8, 1.0),
        grid_default=(0.8, 1.0),
    ),
    ModelParameter(
        name="colsample_bytree",
        label="Column sample fraction",
        kind="float",
        default=1.0,
        minimum=0.1,
        maximum=1.0,
        step=0.05,
        grid_options=(0.6, 0.8, 1.0),
        grid_default=(0.8, 1.0),
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


VOTING_REGRESSOR_MODELS: tuple[VotingEstimatorDefinition, ...] = (
    VotingEstimatorDefinition(
        key="linear_regression",
        label="Linear Regression",
        create_estimator=_create_linear_regression,
        parameters=LINEAR_REGRESSION_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="ridge",
        label="Ridge Regression",
        create_estimator=_create_ridge,
        parameters=REGULARIZED_LINEAR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="lasso",
        label="Lasso Regression",
        create_estimator=_create_lasso,
        parameters=REGULARIZED_LINEAR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="elastic_net",
        label="Elastic Net",
        create_estimator=_create_elastic_net,
        parameters=ELASTIC_NET_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="lars",
        label="Least Angle Regression",
        create_estimator=_create_lars,
        parameters=LARS_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="lasso_lars",
        label="Lasso Lars",
        create_estimator=_create_lasso_lars,
        parameters=LASSO_LARS_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="bayesian_ridge",
        label="Bayesian Ridge",
        create_estimator=_create_bayesian_ridge,
        parameters=BAYESIAN_RIDGE_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="huber_regressor",
        label="Huber Regressor",
        create_estimator=_create_huber_regressor,
        parameters=HUBER_REGRESSOR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="sgd_regressor",
        label="SGD Regressor",
        create_estimator=_create_sgd_regressor,
        parameters=SGD_REGRESSOR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="quantile_regressor",
        label="Quantile Regressor",
        create_estimator=_create_quantile_regressor,
        parameters=QUANTILE_REGRESSOR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="tweedie_regressor",
        label="Tweedie Regressor",
        create_estimator=_create_tweedie_regressor,
        parameters=TWEEDIE_REGRESSOR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="decision_tree",
        label="Decision Tree Regressor",
        create_estimator=_create_decision_tree,
        parameters=TREE_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="extra_tree",
        label="Extra Tree Regressor",
        create_estimator=_create_extra_tree,
        parameters=TREE_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="random_forest",
        label="Random Forest Regressor",
        create_estimator=_create_random_forest,
        parameters=RANDOM_FOREST_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="extra_trees",
        label="Extra Trees Regressor",
        create_estimator=_create_extra_trees,
        parameters=EXTRA_TREES_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="gradient_boosting",
        label="Gradient Boosting Regressor",
        create_estimator=_create_gradient_boosting,
        parameters=GRADIENT_BOOSTING_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="hist_gradient_boosting",
        label="Histogram Gradient Boosting Regressor",
        create_estimator=_create_hist_gradient_boosting,
        parameters=HIST_GRADIENT_BOOSTING_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="adaboost",
        label="AdaBoost Regressor",
        create_estimator=_create_adaboost,
        parameters=ADABOOST_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="bagging",
        label="Bagging Regressor",
        create_estimator=_create_bagging,
        parameters=BAGGING_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="knn",
        label="K-Neighbors Regressor",
        create_estimator=_create_knn,
        parameters=KNN_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="svm",
        label="Support Vector Regressor",
        create_estimator=_create_svr,
        parameters=SVM_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="nu_svr",
        label="Nu Support Vector Regressor",
        create_estimator=_create_nu_svr,
        parameters=NU_SVR_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="kernel_ridge",
        label="Kernel Ridge Regressor",
        create_estimator=_create_kernel_ridge,
        parameters=KERNEL_RIDGE_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="gaussian_process",
        label="Gaussian Process Regressor",
        create_estimator=_create_gaussian_process,
        parameters=GAUSSIAN_PROCESS_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="mlp",
        label="Multi-layer Perceptron Regressor",
        create_estimator=_create_mlp,
        parameters=MLP_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="pls_regression",
        label="Partial Least Squares Regressor",
        create_estimator=_create_pls_regression,
        parameters=PLS_REGRESSION_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="elm",
        label="Extreme Learning Machine Regressor",
        create_estimator=_create_elm,
        parameters=ELM_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="xgboost",
        label="XGBoost Regressor",
        create_estimator=_create_xgboost,
        parameters=XGBOOST_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="catboost",
        label="CatBoost Regressor",
        create_estimator=_create_catboost,
        parameters=CATBOOST_PARAMETERS,
    ),
    VotingEstimatorDefinition(
        key="lightgbm",
        label="LightGBM Regressor",
        create_estimator=_create_lightgbm,
        parameters=LIGHTGBM_PARAMETERS,
    ),
)


MODEL_REGISTRY: dict[str, ModelDefinition] = {
    model.key: ModelDefinition(
        key=model.key,
        label=model.label,
        create_estimator=model.create_estimator,
        parameters=model.parameters,
    )
    for model in VOTING_REGRESSOR_MODELS
}

MODEL_REGISTRY["voting_regressor"] = ModelDefinition(
    key="voting_regressor",
    label="Voting Regressor",
    create_estimator=_create_voting_regressor,
    parameters=tuple(
        parameter
        for model in VOTING_REGRESSOR_MODELS
        for parameter in _prefixed_parameters(model.key, model.label, model.parameters)
    ),
)


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
