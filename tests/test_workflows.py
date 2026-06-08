import unittest

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline

from mlstudio.backend import (
    ModelConfig,
    PipelineStep,
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
from mlstudio.backend.types import RegressionMetrics


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
        assert result.prediction is not None
        self.assertEqual(result.prediction.processed.preprocessed.columns, ["feature"])
        self.assertEqual(result.prediction.processed.model_input.columns, ["feature"])
        self.assertEqual(
            result.prediction.processed.selected_features,
            ("feature",),
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

    def test_cross_validation_returns_out_of_fold_predictions(self) -> None:
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

        self.assertIsInstance(result.metrics, RegressionMetrics)
        self.assertIsNotNone(result.prediction)
        assert result.prediction is not None
        self.assertEqual(result.prediction.data.height, self.data.height)
        self.assertEqual(result.prediction.data.columns, ["Real", "Prediction"])
        self.assertEqual(
            result.prediction.data["Real"].to_list(),
            self.data["target"].to_list(),
        )

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

    def test_optional_pipeline_step_is_fitted_between_preprocessing_and_model(
        self,
    ) -> None:
        data = self.data.with_columns(
            pl.Series("noise", [float(value % 3) for value in range(20)])
        )
        result = train(
            training_data=data,
            test_data=None,
            features=["feature", "noise"],
            target="target",
            preprocessing=get_preprocessing_data(data.select("feature", "noise")),
            model=self.model,
            row_selection="Random percent",
            training_percent=100,
            pipeline_steps=(
                PipelineStep(
                    "test_selection",
                    SelectKBest(score_func=f_regression, k=1),
                ),
            ),
        )

        pipeline = result.artifact.pipeline
        assert isinstance(pipeline, Pipeline)
        selector = pipeline.named_steps["test_selection"]
        self.assertEqual(selector.get_support().sum(), 1)

        prediction = predict(result.artifact, data.tail(4))
        self.assertEqual(prediction.processed.preprocessed.width, 2)
        self.assertEqual(prediction.processed.model_input.width, 1)
        self.assertEqual(len(prediction.processed.selected_features), 1)

    def test_processed_data_contains_encoded_feature_names(self) -> None:
        data = pl.DataFrame(
            {
                "number": [float(value) for value in range(20)],
                "category": ["a", "b"] * 10,
                "target": [float(value * 2) for value in range(20)],
            }
        )
        result = train(
            training_data=data,
            test_data=data.tail(4),
            features=["number", "category"],
            target="target",
            preprocessing=get_preprocessing_data(data.select("number", "category")),
            model=self.model,
            row_selection="Random percent",
            training_percent=100,
        )

        assert result.prediction is not None
        encoded_columns = result.prediction.processed.preprocessed.columns
        self.assertIn("category_a", encoded_columns)
        self.assertIn("category_b", encoded_columns)
        self.assertIn("number", encoded_columns)


if __name__ == "__main__":
    unittest.main()
