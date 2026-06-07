import unittest

import numpy as np
import polars as pl

from mlstudio.ml.data import (
    create_preprocessing_transformer,
    get_preprocessing_data,
)
from mlstudio.ml.modeling import (
    ModelConfig,
    calculate_metrics,
    create_artifact,
    create_estimator,
    deserialize_artifact,
    prediction_frame,
    select_training_rows,
    serialize_artifact,
    split_validation_data,
    validate_feature_columns,
    validate_feature_schema,
)
from mlstudio.ml.models import get_model_definitions


class ModelingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pl.DataFrame(
            {
                "row": list(range(10)),
                "feature": [float(value) for value in range(10)],
                "target": [float(value * 2) for value in range(10)],
            }
        )

    def test_last_percent_selects_final_rows(self) -> None:
        selected = select_training_rows(self.df, "Last percent", 30)
        self.assertEqual(selected["row"].to_list(), [7, 8, 9])

    def test_random_percent_is_reproducible(self) -> None:
        first = select_training_rows(self.df, "Random percent", 40)
        second = select_training_rows(self.df, "Random percent", 40)
        self.assertEqual(first["row"].to_list(), second["row"].to_list())
        self.assertEqual(first.height, 4)

    def test_last_validation_split_preserves_order(self) -> None:
        train, validation = split_validation_data(
            self.df,
            "Last split",
            20,
        )
        self.assertEqual(train["row"].to_list(), list(range(8)))
        self.assertEqual(validation["row"].to_list(), [8, 9])

    def test_metrics_exclude_zero_targets_from_mape(self) -> None:
        metrics = calculate_metrics(
            pl.Series([0.0, 10.0, 20.0]),
            np.array([5.0, 8.0, 22.0]),
        )
        self.assertAlmostEqual(metrics.mape or 0, 15.0)

    def test_all_zero_targets_make_mape_unavailable(self) -> None:
        metrics = calculate_metrics(
            pl.Series([0.0, 0.0]),
            np.array([1.0, 2.0]),
        )
        self.assertIsNone(metrics.mape)

    def test_pipeline_and_artifact_round_trip(self) -> None:
        features = ["feature"]
        preprocessing = get_preprocessing_data(self.df.select(features))
        transformer = create_preprocessing_transformer(preprocessing)
        model = get_model_definitions()["ridge"]
        config = ModelConfig(
            definition=model,
            parameters={"alpha": 1.0, "fit_intercept": True},
            use_grid_search=False,
            param_grid={},
            cv=5,
        )
        estimator = create_estimator(transformer, config)
        estimator.fit(self.df.select(features), self.df["target"])
        artifact = create_artifact(
            estimator,
            self.df,
            features,
            "target",
            model.label,
        )

        restored = deserialize_artifact(serialize_artifact(artifact))
        predictions = restored.pipeline.predict(self.df.select(features))

        self.assertEqual(restored.features, ("feature",))
        self.assertEqual(len(predictions), self.df.height)

    def test_feature_validation_reports_missing_and_incompatible_columns(self) -> None:
        with self.assertRaisesRegex(ValueError, "Missing required"):
            validate_feature_columns(self.df, ["missing"])
        with self.assertRaisesRegex(ValueError, "Incompatible feature types"):
            validate_feature_schema(
                pl.DataFrame({"feature": ["a", "b"]}),
                {"feature": "Float64"},
            )

    def test_prediction_frame_contains_only_real_and_prediction(self) -> None:
        results = prediction_frame(
            self.df,
            np.arange(self.df.height),
            "target",
        )
        self.assertEqual(results.columns, ["Real", "Prediction"])

    def test_prediction_frame_without_target_contains_only_prediction(self) -> None:
        results = prediction_frame(self.df, np.arange(self.df.height))
        self.assertEqual(results.columns, ["Prediction"])


if __name__ == "__main__":
    unittest.main()
