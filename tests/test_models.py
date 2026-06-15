import unittest

from catboost import CatBoostRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor

from mlstudio.backend.elm import ELMRegressor
from mlstudio.backend.models import (
    get_model_definitions,
    is_parameter_visible,
)


class ModelDefinitionTests(unittest.TestCase):
    def test_new_regressors_are_registered(self) -> None:
        definitions = get_model_definitions()

        self.assertIsInstance(self._create_default("svm"), SVR)
        self.assertIsInstance(self._create_default("elm"), ELMRegressor)
        self.assertIsInstance(self._create_default("xgboost"), XGBRegressor)
        self.assertIsInstance(
            self._create_default("catboost"),
            CatBoostRegressor,
        )
        self.assertTrue({"svm", "elm", "xgboost", "catboost"}.issubset(definitions))

    def test_svm_parameters_follow_kernel_dependencies(self) -> None:
        parameters = {
            parameter.name: parameter
            for parameter in get_model_definitions()["svm"].parameters
        }

        self.assertFalse(is_parameter_visible(parameters["degree"], {"kernel": "rbf"}))
        self.assertTrue(is_parameter_visible(parameters["degree"], {"kernel": "poly"}))
        self.assertFalse(
            is_parameter_visible(parameters["gamma"], {"kernel": "linear"})
        )
        self.assertTrue(
            is_parameter_visible(
                parameters["gamma"],
                {"kernel": ["linear", "rbf"]},
            )
        )
        self.assertTrue(
            is_parameter_visible(
                parameters["coef0"],
                {"kernel": ["rbf", "sigmoid"]},
            )
        )

    def _create_default(self, key: str):
        definition = get_model_definitions()[key]
        parameters = {
            parameter.name: parameter.default
            for parameter in definition.parameters
            if is_parameter_visible(
                parameter,
                {
                    candidate.name: candidate.default
                    for candidate in definition.parameters
                },
            )
        }
        return definition.create_estimator(parameters)


if __name__ == "__main__":
    unittest.main()
