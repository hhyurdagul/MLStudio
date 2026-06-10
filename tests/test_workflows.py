import unittest

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import VotingRegressor
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
from mlstudio.plugins.lookback import create_lookback_wrapper


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

    def test_voting_regressor_is_registered_and_trainable(self) -> None:
        definition = get_model_definitions()["voting_regressor"]
        parameters = {
            parameter.name: parameter.default
            for parameter in definition.parameters
        }
        estimator = definition.create_estimator(parameters)

        assert isinstance(estimator, VotingRegressor)
        self.assertEqual(
            [parameter.name for parameter in definition.parameters],
            [
                "random_forest__n_estimators",
                "random_forest__max_depth",
                "random_forest__min_samples_split",
                "gradient_boosting__n_estimators",
                "gradient_boosting__learning_rate",
                "gradient_boosting__max_depth",
                "ridge__alpha",
                "ridge__fit_intercept",
            ],
        )
        estimator.fit(
            self.data.select("feature").to_numpy(),
            self.data["target"].to_numpy(),
        )
        self.assertEqual(estimator.predict([[20.0]]).shape, (1,))

    def test_voting_regressor_grid_search_tunes_nested_estimators(self) -> None:
        definition = get_model_definitions()["voting_regressor"]
        model = ModelConfig(
            definition=definition,
            parameters={
                parameter.name: parameter.default
                for parameter in definition.parameters
            },
            use_grid_search=True,
            param_grid={"model__ridge__alpha": [0.1, 1.0]},
            cv=3,
        )
        result = train(
            training_data=self.data,
            test_data=None,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=model,
            row_selection="Random percent",
            training_percent=100,
        )

        assert result.grid_search is not None
        self.assertIn(
            "model__ridge__alpha",
            result.grid_search.best_parameters,
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

    def test_lookback_lags_are_added_after_feature_selection(self) -> None:
        data = self.data.with_columns(
            pl.Series("noise", [float(value % 3) for value in range(20)])
        )
        result = train(
            training_data=data,
            test_data=data.tail(4),
            features=["feature", "noise"],
            target="target",
            preprocessing=get_preprocessing_data(data.select("feature", "noise")),
            model=self.model,
            row_selection="Last percent",
            training_percent=80,
            pipeline_steps=(
                PipelineStep(
                    "test_selection",
                    SelectKBest(score_func=f_regression, k=1),
                ),
            ),
            estimator_wrapper=create_lookback_wrapper(3),
        )

        assert result.prediction is not None
        self.assertEqual(
            result.prediction.processed.selected_features[-3:],
            ("target_lag_1", "target_lag_2", "target_lag_3"),
        )
        self.assertEqual(result.prediction.processed.model_input.width, 4)

    def test_lookback_artifact_round_trip_resets_prediction_history(self) -> None:
        result = train(
            training_data=self.data,
            test_data=None,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            row_selection="Last percent",
            training_percent=100,
            estimator_wrapper=create_lookback_wrapper(2),
        )
        artifact = deserialize_artifact(serialize_artifact(result.artifact))
        prediction_data = self.data.tail(3).select("feature")

        first = predict(artifact, prediction_data)
        second = predict(artifact, prediction_data)

        self.assertEqual(
            first.data["Prediction"].to_list(),
            second.data["Prediction"].to_list(),
        )

    def test_lookback_rejects_random_training_rows(self) -> None:
        wrapper = create_lookback_wrapper(2)
        with self.assertRaisesRegex(ValueError, "last contiguous"):
            train(
                training_data=self.data,
                test_data=None,
                features=["feature"],
                target="target",
                preprocessing=self.preprocessing,
                model=self.model,
                row_selection="Random percent",
                training_percent=100,
                estimator_wrapper=wrapper,
            )

    def test_lookback_cross_validation_uses_time_series_folds(self) -> None:
        result = validate(
            data=self.data,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            strategy="Cross-validation",
            validation_percent=20,
            folds=4,
            estimator_wrapper=create_lookback_wrapper(2),
        )

        assert result.prediction is not None
        self.assertEqual(result.prediction.data.height, 16)
        self.assertEqual(
            result.prediction.data["Real"].to_list(),
            self.data.tail(16)["target"].to_list(),
        )

    def test_lookback_grid_search_tunes_wrapped_estimator(self) -> None:
        model = ModelConfig(
            definition=self.model.definition,
            parameters=self.model.parameters,
            use_grid_search=True,
            param_grid={"model__alpha": [0.1, 1.0]},
            cv=3,
        )
        result = train(
            training_data=self.data,
            test_data=None,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=model,
            row_selection="Last percent",
            training_percent=100,
            estimator_wrapper=create_lookback_wrapper(2),
        )

        assert result.grid_search is not None
        self.assertIn(
            "model__estimator__alpha",
            result.grid_search.best_parameters,
        )

    def test_target_processing_inverse_transforms_predictions(self) -> None:
        result = train(
            training_data=self.data,
            test_data=self.data.tail(3),
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            row_selection="Last percent",
            training_percent=100,
            target_processing="StandardScaler",
        )

        assert result.prediction is not None
        predictions = result.prediction.data["Prediction"].to_numpy()
        self.assertGreater(float(predictions.max()), 1.0)

    def test_target_processing_is_applied_to_lookback_lags(self) -> None:
        result = train(
            training_data=self.data,
            test_data=self.data.tail(2),
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=self.model,
            row_selection="Last percent",
            training_percent=100,
            target_processing="MinMaxScaler",
            estimator_wrapper=create_lookback_wrapper(2),
        )

        assert result.prediction is not None
        lag_values = result.prediction.processed.model_input.select(
            "target_lag_1",
            "target_lag_2",
        ).to_numpy()
        self.assertTrue(np.all((lag_values >= 0.0) & (lag_values <= 1.0)))
        self.assertGreater(
            float(result.prediction.data["Prediction"].to_numpy().max()),
            1.0,
        )

    def test_grid_search_routes_through_target_processing_and_lookback(
        self,
    ) -> None:
        model = ModelConfig(
            definition=self.model.definition,
            parameters=self.model.parameters,
            use_grid_search=True,
            param_grid={"model__alpha": [0.1, 1.0]},
            cv=3,
        )
        result = train(
            training_data=self.data,
            test_data=None,
            features=["feature"],
            target="target",
            preprocessing=self.preprocessing,
            model=model,
            row_selection="Last percent",
            training_percent=100,
            target_processing="StandardScaler",
            estimator_wrapper=create_lookback_wrapper(2),
        )

        assert result.grid_search is not None
        self.assertIn(
            "model__regressor__estimator__alpha",
            result.grid_search.best_parameters,
        )


if __name__ == "__main__":
    unittest.main()
