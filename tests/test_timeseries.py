import unittest

import numpy as np
import polars as pl
import torch

from mlstudio.backend.timeseries import (
    TIMESERIES_MODEL_NAMES,
    TimeSeriesConfig,
    TimeSeriesModelName,
    create_timeseries_model,
    deserialize_timeseries_artifact,
    forecast_timeseries_model,
    serialize_timeseries_artifact,
    train_timeseries_model,
)
from mlstudio.backend.autoregression import select_lags


class TimeSeriesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)

    def setUp(self) -> None:
        self.data = pl.DataFrame(
            {"target": [float(value) for value in range(12)]}
        )

    def config(
        self,
        model_name: TimeSeriesModelName = "MLP",
    ) -> TimeSeriesConfig:
        return TimeSeriesConfig(
            model_name=model_name,
            max_time_lag=3,
            lag_indices=(1, 2, 3),
            layer_sizes=(4, 3),
            learning_rate=0.01,
            epochs=1,
            batch_size=4,
            hidden_activation="ReLU",
            output_activation="Linear",
            target_processing="StandardScaler",
        )

    def test_all_models_produce_one_value_per_window(self) -> None:
        inputs = torch.zeros((2, 3, 1))
        for model_name in TIMESERIES_MODEL_NAMES:
            with self.subTest(model=model_name):
                output = create_timeseries_model(self.config(model_name))(inputs)
                self.assertEqual(tuple(output.shape), (2, 1))

    def test_all_models_complete_a_training_epoch(self) -> None:
        for model_name in TIMESERIES_MODEL_NAMES:
            with self.subTest(model=model_name):
                result = train_timeseries_model(
                    self.data,
                    "target",
                    self.config(model_name),
                )
                self.assertEqual(result.trained_windows, 9)
                self.assertTrue(np.isfinite(result.losses[0]))

    def test_training_uses_consecutive_sliding_windows(self) -> None:
        result = train_timeseries_model(
            self.data,
            "target",
            self.config(),
        )

        self.assertEqual(result.trained_windows, 9)
        self.assertEqual(len(result.losses), 1)
        self.assertEqual(result.artifact.target_history, (9.0, 10.0, 11.0))

    def test_training_percent_selects_last_contiguous_rows(self) -> None:
        result = train_timeseries_model(
            self.data,
            "target",
            self.config(),
            training_percent=50,
        )

        self.assertEqual(result.trained_windows, 3)
        self.assertEqual(result.artifact.target_history, (9.0, 10.0, 11.0))

    def test_training_percent_must_leave_more_rows_than_lookback(self) -> None:
        with self.assertRaisesRegex(ValueError, "Selected training data"):
            train_timeseries_model(
                self.data,
                "target",
                self.config(),
                training_percent=25,
            )

    def test_backtest_is_recursive_and_does_not_consume_actual_values(self) -> None:
        result = train_timeseries_model(self.data, "target", self.config())
        low_actuals = pl.DataFrame({"target": [0.0, 0.0, 0.0]})
        high_actuals = pl.DataFrame({"target": [100.0, 200.0, 300.0]})

        low = forecast_timeseries_model(result.artifact, 3, low_actuals)
        high = forecast_timeseries_model(result.artifact, 3, high_actuals)

        np.testing.assert_allclose(
            low.prediction.data["Prediction"].to_numpy(),
            high.prediction.data["Prediction"].to_numpy(),
        )

    def test_artifact_round_trip_preserves_forecasts(self) -> None:
        result = train_timeseries_model(self.data, "target", self.config("GRU"))
        restored = deserialize_timeseries_artifact(
            serialize_timeseries_artifact(result.artifact)
        )

        original = forecast_timeseries_model(result.artifact, 4)
        loaded = forecast_timeseries_model(restored, 4)

        np.testing.assert_allclose(
            original.prediction.data["Prediction"].to_numpy(),
            loaded.prediction.data["Prediction"].to_numpy(),
        )

    def test_rejects_short_or_invalid_target_series(self) -> None:
        with self.assertRaisesRegex(ValueError, "more rows than the max time lag"):
            train_timeseries_model(
                self.data.head(3),
                "target",
                self.config(),
            )
        with self.assertRaisesRegex(ValueError, "must be numeric"):
            train_timeseries_model(
                pl.DataFrame({"target": ["a", "b", "c", "d"]}),
                "target",
                self.config(),
            )

    def test_actual_rows_must_match_horizon(self) -> None:
        result = train_timeseries_model(self.data, "target", self.config())

        with self.assertRaisesRegex(ValueError, "match the forecast horizon"):
            forecast_timeseries_model(
                result.artifact,
                3,
                pl.DataFrame({"target": [1.0, 2.0]}),
            )

    def test_rejects_malformed_artifact(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a valid"):
            deserialize_timeseries_artifact(b"not a pytorch artifact")

    def test_sparse_lag_indices_control_model_inputs_and_forecast_history(
        self,
    ) -> None:
        config = TimeSeriesConfig(
            model_name="MLP",
            max_time_lag=5,
            lag_indices=(1, 3, 5),
            layer_sizes=(4,),
            epochs=1,
            batch_size=4,
        )
        result = train_timeseries_model(self.data, "target", config)

        self.assertEqual(result.trained_windows, 7)
        self.assertEqual(
            result.artifact.target_history,
            (7.0, 8.0, 9.0, 10.0, 11.0),
        )
        self.assertEqual(
            forecast_timeseries_model(result.artifact, 2).prediction.data.height,
            2,
        )

    def test_lag_selection_modes(self) -> None:
        values = np.asarray([1.0, 2.0, 4.0, 3.0, 5.0, 7.0, 6.0, 8.0])

        self.assertEqual(select_lags(values, 4, "All"), (1, 2, 3, 4))
        self.assertEqual(
            select_lags(values, 4, "Manual", indices=(4, 2, 2)),
            (2, 4),
        )
        best = select_lags(values, 4, "Best N ACF", count=2)
        self.assertEqual(len(best), 2)
        thresholded = select_lags(
            values,
            4,
            "ACF threshold",
            threshold=0.0,
        )
        self.assertEqual(thresholded, (1, 2, 3, 4))


if __name__ == "__main__":
    unittest.main()
