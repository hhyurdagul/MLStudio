import unittest

import numpy as np
import polars as pl
import torch

from mlstudio.backend.deep_learning import (
    DEEP_MODEL_NAMES,
    DeepLearningConfig,
    DeepModelName,
    create_deep_model,
    deserialize_deep_artifact,
    forecast_deep_model,
    serialize_deep_artifact,
    train_deep_model,
)


class DeepLearningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.data = pl.DataFrame(
            {"target": [float(value) for value in range(12)]}
        )

    def config(self, model_name: DeepModelName = "MLP") -> DeepLearningConfig:
        return DeepLearningConfig(
            model_name=model_name,
            lookback=3,
            neurons=4,
            layers=2,
            learning_rate=0.01,
            epochs=1,
            batch_size=4,
            hidden_activation="ReLU",
            output_activation="Linear",
            target_processing="StandardScaler",
        )

    def test_all_models_produce_one_value_per_window(self) -> None:
        inputs = torch.zeros((2, 3, 1))
        for model_name in DEEP_MODEL_NAMES:
            with self.subTest(model=model_name):
                output = create_deep_model(self.config(model_name))(inputs)
                self.assertEqual(tuple(output.shape), (2, 1))

    def test_all_models_complete_a_training_epoch(self) -> None:
        for model_name in DEEP_MODEL_NAMES:
            with self.subTest(model=model_name):
                result = train_deep_model(
                    self.data,
                    "target",
                    self.config(model_name),
                )
                self.assertEqual(result.trained_windows, 9)
                self.assertTrue(np.isfinite(result.losses[0]))

    def test_training_uses_consecutive_sliding_windows(self) -> None:
        result = train_deep_model(
            self.data,
            "target",
            self.config(),
        )

        self.assertEqual(result.trained_windows, 9)
        self.assertEqual(len(result.losses), 1)
        self.assertEqual(result.artifact.target_history, (9.0, 10.0, 11.0))

    def test_training_percent_selects_last_contiguous_rows(self) -> None:
        result = train_deep_model(
            self.data,
            "target",
            self.config(),
            training_percent=50,
        )

        self.assertEqual(result.trained_windows, 3)
        self.assertEqual(result.artifact.target_history, (9.0, 10.0, 11.0))

    def test_training_percent_must_leave_more_rows_than_lookback(self) -> None:
        with self.assertRaisesRegex(ValueError, "Selected training data"):
            train_deep_model(
                self.data,
                "target",
                self.config(),
                training_percent=25,
            )

    def test_backtest_is_recursive_and_does_not_consume_actual_values(self) -> None:
        result = train_deep_model(self.data, "target", self.config())
        low_actuals = pl.DataFrame({"target": [0.0, 0.0, 0.0]})
        high_actuals = pl.DataFrame({"target": [100.0, 200.0, 300.0]})

        low = forecast_deep_model(result.artifact, 3, low_actuals)
        high = forecast_deep_model(result.artifact, 3, high_actuals)

        np.testing.assert_allclose(
            low.prediction.data["Prediction"].to_numpy(),
            high.prediction.data["Prediction"].to_numpy(),
        )

    def test_artifact_round_trip_preserves_forecasts(self) -> None:
        result = train_deep_model(self.data, "target", self.config("GRU"))
        restored = deserialize_deep_artifact(
            serialize_deep_artifact(result.artifact)
        )

        original = forecast_deep_model(result.artifact, 4)
        loaded = forecast_deep_model(restored, 4)

        np.testing.assert_allclose(
            original.prediction.data["Prediction"].to_numpy(),
            loaded.prediction.data["Prediction"].to_numpy(),
        )

    def test_rejects_short_or_invalid_target_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "more rows than the lookback"):
            train_deep_model(
                self.data.head(3),
                "target",
                self.config(),
            )
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            train_deep_model(
                pl.DataFrame({"target": ["a", "b", "c", "d"]}),
                "target",
                self.config(),
            )

    def test_actual_rows_must_match_horizon(self) -> None:
        result = train_deep_model(self.data, "target", self.config())

        with self.assertRaisesRegex(ValueError, "match the forecast horizon"):
            forecast_deep_model(
                result.artifact,
                3,
                pl.DataFrame({"target": [1.0, 2.0]}),
            )

    def test_rejects_malformed_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a valid"):
            deserialize_deep_artifact(b"not a pytorch artifact")


if __name__ == "__main__":
    unittest.main()
