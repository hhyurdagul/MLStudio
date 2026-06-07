import unittest

import numpy as np
import polars as pl

from mlstudio.backend import (
    ModelConfig,
    deserialize_artifact,
    get_model_definitions,
    get_preprocessing_data,
    predict,
    serialize_artifact,
    train,
    validate,
)
from mlstudio.backend.evaluation import (
    calculate_metrics,
    select_training_rows,
    split_validation_data,
)
from mlstudio.backend.types import CrossValidationMetrics, RegressionMetrics


class WorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = pl.DataFrame(
            {
                "row": list(range(20)),
                "feature": [float(value) for value in range(20)],
                "target": [float(value * 2) for value in range(20)],
            }
        )
        model = get_model_definitions()["ridge"]
        self.model = ModelConfig(
            definition=model,
            parameters={"alpha": 1.0, "fit_intercept": True},
            use_grid_search=False,
            param_grid={},
            cv=5,
        )
        self.preprocessing = get_preprocessing_data(self.data.select("feature"))

    def test_last_percent_selects_final_rows(self) -> None:
        selected = select_training_rows(self.data, "Last percent", 25)
        self.assertEqual(selected["row"].to_list(), [15, 16, 17, 18, 19])

    def test_random_percent_is_reproducible(self) -> None:
        first = select_training_rows(self.data, "Random percent", 40)
        second = select_training_rows(self.data, "Random percent", 40)
        self.assertEqual(first["row"].to_list(), second["row"].to_list())

    def test_last_validation_split_preserves_order(self) -> None:
        training, validation_data = split_validation_data(
            self.data,
            "Last split",
            20,
        )
        self.assertEqual(training["row"].to_list(), list(range(16)))
        self.assertEqual(validation_data["row"].to_list(), [16, 17, 18, 19])

    def test_metrics_exclude_zero_targets_from_mape(self) -> None:
        metrics = calculate_metrics(
            pl.Series([0.0, 10.0, 20.0]),
            np.array([5.0, 8.0, 22.0]),
        )
        self.assertAlmostEqual(metrics.mape or 0, 15.0)

    def test_training_returns_artifact_and_test_predictions(self) -> None:
        result = train(
            training_data=self.data,
            test_data=self.data.tail(4),
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            row_selection="Last percent",
            training_percent=50,
        )

        self.assertEqual(result.trained_rows, 10)
        self.assertIsNotNone(result.prediction)
        self.assertEqual(
            result.prediction.data.columns if result.prediction else [],
            ["Real", "Prediction"],
        )

    def test_artifact_round_trip_supports_unlabeled_prediction(self) -> None:
        result = train(
            training_data=self.data,
            test_data=None,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            row_selection="Random percent",
            training_percent=100,
        )
        artifact = deserialize_artifact(serialize_artifact(result.artifact))
        prediction = predict(artifact, self.data.select("feature"))

        self.assertEqual(prediction.data.columns, ["Prediction"])
        self.assertIsNone(prediction.metrics)

    def test_holdout_validation_returns_metrics_and_predictions(self) -> None:
        result = validate(
            data=self.data,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            strategy="Last split",
            validation_percent=20,
            folds=5,
        )

        self.assertIsInstance(result.metrics, RegressionMetrics)
        self.assertIsNotNone(result.prediction)

    def test_cross_validation_returns_aggregate_metrics(self) -> None:
        result = validate(
            data=self.data,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            strategy="Cross-validation",
            validation_percent=20,
            folds=5,
        )

        self.assertIsInstance(result.metrics, CrossValidationMetrics)
        self.assertIsNone(result.prediction)

    def test_prediction_rejects_missing_features(self) -> None:
        result = train(
            training_data=self.data,
            test_data=None,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            row_selection="Random percent",
            training_percent=100,
        )

        with self.assertRaisesRegex(ValueError, "Missing required feature"):
            predict(result.artifact, self.data.select("target"))


if __name__ == "__main__":
    unittest.main()
