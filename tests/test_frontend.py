import importlib
import unittest
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import polars as pl

from mlstudio.frontend.components import results

supervised_test_page = importlib.import_module("mlstudio.frontend.pages.test")
supervised_training_page = importlib.import_module(
    "mlstudio.frontend.pages.training"
)
timeseries_page = importlib.import_module("mlstudio.frontend.pages.timeseries")


class PredictionChartTests(unittest.TestCase):
    def test_row_axis_starts_at_one_and_uses_integer_ticks(self) -> None:
        streamlit = MagicMock()
        streamlit.tabs.return_value = (nullcontext(), nullcontext())

        with patch.object(results, "st", streamlit):
            results.render_predictions(
                pl.DataFrame(
                    {
                        "Real": [1.0, 2.0, 3.0, 4.0],
                        "Prediction": [1.1, 2.1, 3.1, 4.1],
                    }
                )
            )

        chart = streamlit.altair_chart.call_args.args[0]
        specification = chart.to_dict()
        x_encoding = specification["encoding"]["x"]
        rows = next(iter(specification["datasets"].values()))
        self.assertEqual(sorted({row["Row"] for row in rows}), [1, 2, 3, 4])
        self.assertEqual(x_encoding["axis"]["format"], "d")
        self.assertEqual(x_encoding["axis"]["tickMinStep"], 1)
        self.assertEqual(x_encoding["scale"]["domainMin"], 1)
        self.assertFalse(x_encoding["scale"]["nice"])


class ForecastCountTests(unittest.TestCase):
    def test_supervised_predictions_are_limited_to_selected_count(self) -> None:
        streamlit = MagicMock()
        streamlit.columns.return_value = (nullcontext(), nullcontext())
        streamlit.file_uploader.return_value.getvalue.return_value = b"model"
        streamlit.number_input.return_value = 2
        streamlit.button.return_value = True
        streamlit.spinner.return_value = nullcontext()
        data = pl.DataFrame({"feature": [10, 20, 30, 40]})
        artifact = SimpleNamespace(
            model_label="Model",
            target="target",
            features=("feature",),
        )
        prediction_result = SimpleNamespace(metrics=None, data=pl.DataFrame())
        test_result = SimpleNamespace(
            processed=SimpleNamespace(),
            prediction=prediction_result,
        )

        with (
            patch.object(supervised_test_page, "st", streamlit),
            patch.object(
                supervised_test_page,
                "render_dataset_selector",
                return_value=data,
            ),
            patch.object(
                supervised_test_page,
                "deserialize_artifact",
                return_value=artifact,
            ),
            patch.object(
                supervised_test_page,
                "predict",
                return_value=test_result,
            ) as predict,
            patch.object(supervised_test_page, "render_processed_data"),
            patch.object(supervised_test_page, "render_predictions"),
        ):
            supervised_test_page.render_test_page()

        predicted_data = predict.call_args.args[1]
        self.assertEqual(predicted_data.to_dict(as_series=False), {"feature": [10, 20]})
        self.assertEqual(streamlit.number_input.call_args.kwargs["max_value"], 4)
        self.assertEqual(streamlit.number_input.call_args.kwargs["value"], 4)

    def test_timeseries_forecasts_are_limited_to_selected_count(self) -> None:
        streamlit = MagicMock()
        streamlit.file_uploader.return_value.getvalue.return_value = b"model"
        streamlit.number_input.return_value = 2
        streamlit.button.return_value = True
        streamlit.spinner.return_value = nullcontext()
        actual_data = pl.DataFrame({"target": [1.0, 2.0, 3.0, 4.0]})
        artifact = SimpleNamespace(
            target="target",
            config=SimpleNamespace(
                model_name="MLP",
                max_time_lag=2,
                lag_indices=(1, 2),
                layer_sizes=(4,),
            ),
        )
        result = SimpleNamespace(
            prediction=SimpleNamespace(metrics=None, data=pl.DataFrame())
        )

        with (
            patch.object(timeseries_page, "st", streamlit),
            patch.object(
                timeseries_page,
                "deserialize_timeseries_artifact",
                return_value=artifact,
            ),
            patch.object(
                timeseries_page,
                "render_dataset_selector",
                return_value=actual_data,
            ),
            patch.object(
                timeseries_page,
                "forecast_timeseries_model",
                return_value=result,
            ) as forecast,
            patch.object(timeseries_page, "render_predictions"),
        ):
            timeseries_page._render_test()

        forecast_actuals = forecast.call_args.args[2]
        self.assertEqual(
            forecast_actuals.to_dict(as_series=False),
            {"target": [1.0, 2.0]},
        )
        self.assertEqual(streamlit.number_input.call_args.args[0], "Forecast count")
        self.assertEqual(streamlit.number_input.call_args.kwargs["max_value"], 4)
        self.assertEqual(streamlit.number_input.call_args.kwargs["value"], 4)

    def test_supervised_training_predicts_after_training_with_selected_count(
        self,
    ) -> None:
        streamlit = MagicMock()
        selection_column = MagicMock()
        percent_column = MagicMock()
        streamlit.columns.side_effect = [
            (nullcontext(), nullcontext()),
            (selection_column, percent_column),
        ]
        streamlit.expander.return_value = nullcontext()
        streamlit.number_input.return_value = 2
        streamlit.button.return_value = True
        streamlit.spinner.return_value = nullcontext()
        streamlit.session_state = {}
        selection_column.selectbox.return_value = "Random percent"
        percent_column.slider.return_value = 100
        training_data = pl.DataFrame(
            {"feature": [1, 2, 3, 4], "target": [2, 4, 6, 8]}
        )
        test_data = pl.DataFrame(
            {"feature": [10, 20, 30, 40], "target": [20, 40, 60, 80]}
        )
        model = SimpleNamespace(definition=SimpleNamespace(label="Model"))
        artifact = SimpleNamespace()
        training_result = SimpleNamespace(
            trained_rows=4,
            grid_search=None,
            artifact=artifact,
            processed=SimpleNamespace(),
        )
        metrics = SimpleNamespace()
        prediction_result = SimpleNamespace(
            processed=SimpleNamespace(),
            prediction=SimpleNamespace(
                metrics=metrics,
                data=pl.DataFrame(),
            ),
        )

        with (
            patch.object(supervised_training_page, "st", streamlit),
            patch.object(
                supervised_training_page,
                "render_dataset_selector",
                side_effect=[training_data, test_data],
            ),
            patch.object(
                supervised_training_page,
                "render_feature_target_selector",
                return_value=(["feature"], "target"),
            ),
            patch.object(supervised_training_page, "render_data_preview"),
            patch.object(
                supervised_training_page,
                "get_preprocessing_data",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                supervised_training_page,
                "render_preprocessing_config",
                return_value=SimpleNamespace(),
            ),
            patch.object(
                supervised_training_page,
                "render_target_processing",
                return_value="None",
            ),
            patch.object(
                supervised_training_page,
                "render_model_config",
                return_value=(model, True),
            ),
            patch.object(
                supervised_training_page,
                "train",
                return_value=training_result,
            ) as train,
            patch.object(
                supervised_training_page,
                "predict",
                return_value=prediction_result,
            ) as predict,
            patch.object(
                supervised_training_page,
                "serialize_artifact",
                return_value=b"model",
            ),
            patch.object(supervised_training_page, "render_grid_search"),
            patch.object(supervised_training_page, "render_processed_data"),
            patch.object(
                supervised_training_page,
                "render_metrics",
            ) as render_metrics,
            patch.object(supervised_training_page, "render_predictions"),
        ):
            supervised_training_page.render_training_page()

        self.assertIsNone(train.call_args.args[2])
        self.assertIs(predict.call_args.args[0], artifact)
        predicted_data = predict.call_args.args[1]
        self.assertEqual(
            predicted_data.to_dict(as_series=False),
            {"feature": [10, 20], "target": [20, 40]},
        )
        render_metrics.assert_called_once_with(metrics)
        self.assertEqual(streamlit.number_input.call_args.args[0], "Prediction count")
        self.assertEqual(streamlit.number_input.call_args.kwargs["value"], 4)

    def test_timeseries_training_forecasts_after_training_with_selected_count(
        self,
    ) -> None:
        streamlit = MagicMock()
        streamlit.columns.side_effect = [
            (nullcontext(), nullcontext()),
            (nullcontext(), nullcontext()),
        ]
        streamlit.expander.return_value = nullcontext()
        streamlit.selectbox.return_value = "target"
        streamlit.slider.return_value = 100
        streamlit.number_input.return_value = 2
        streamlit.button.return_value = True
        streamlit.spinner.return_value = nullcontext()
        streamlit.session_state = {}
        training_data = pl.DataFrame(
            {"target": [float(value) for value in range(12)]}
        )
        backtest_data = pl.DataFrame({"target": [12.0, 13.0, 14.0, 15.0]})
        config = SimpleNamespace(model_name="MLP", output_activation="Linear")
        artifact = SimpleNamespace()
        training_result = SimpleNamespace(
            artifact=artifact,
            trained_windows=10,
            losses=(1.0,),
        )
        metrics = SimpleNamespace()
        forecast_result = SimpleNamespace(
            prediction=SimpleNamespace(
                metrics=metrics,
                data=pl.DataFrame(),
            )
        )

        with (
            patch.object(timeseries_page, "st", streamlit),
            patch.object(
                timeseries_page,
                "render_dataset_selector",
                side_effect=[training_data, backtest_data],
            ),
            patch.object(
                timeseries_page,
                "_render_lag_preview",
                return_value=(2, (1, 2), True),
            ),
            patch.object(timeseries_page, "_render_config", return_value=config),
            patch.object(
                timeseries_page,
                "train_timeseries_model",
                return_value=training_result,
            ) as train,
            patch.object(
                timeseries_page,
                "forecast_timeseries_model",
                return_value=forecast_result,
            ) as forecast,
            patch.object(timeseries_page, "_render_loss_history"),
            patch.object(
                timeseries_page,
                "serialize_timeseries_artifact",
                return_value=b"model",
            ),
            patch.object(timeseries_page, "render_metrics") as render_metrics,
            patch.object(timeseries_page, "render_predictions"),
        ):
            timeseries_page._render_training()

        self.assertIsNone(train.call_args.args[3])
        self.assertIs(forecast.call_args.args[0], artifact)
        self.assertEqual(forecast.call_args.args[1], 2)
        forecast_data = forecast.call_args.args[2]
        self.assertEqual(
            forecast_data.to_dict(as_series=False),
            {"target": [12.0, 13.0]},
        )
        render_metrics.assert_called_once_with(metrics)
        self.assertEqual(streamlit.number_input.call_args.args[0], "Forecast count")
        self.assertEqual(streamlit.number_input.call_args.kwargs["value"], 4)


class TimeSeriesConfigTests(unittest.TestCase):
    def test_learning_rate_minimum_and_precision_are_five_decimals(self) -> None:
        streamlit = MagicMock()
        model_column = MagicMock()
        layers_column = MagicMock()
        neuron_column = MagicMock()
        rate_column = MagicMock()
        epochs_column = MagicMock()
        batch_column = MagicMock()
        hidden_column = MagicMock()
        output_column = MagicMock()
        streamlit.columns.side_effect = [
            (model_column, layers_column),
            (neuron_column,),
            (rate_column, epochs_column),
            (batch_column, hidden_column, output_column),
        ]
        model_column.selectbox.return_value = "MLP"
        layers_column.number_input.return_value = 1
        neuron_column.number_input.return_value = 64
        rate_column.number_input.return_value = 0.00001
        epochs_column.number_input.return_value = 10
        batch_column.number_input.return_value = 4
        hidden_column.selectbox.return_value = "ReLU"
        output_column.selectbox.return_value = "Linear"
        streamlit.selectbox.return_value = "None"

        with patch.object(timeseries_page, "st", streamlit):
            config = timeseries_page._render_config(2, (1, 2))

        self.assertEqual(config.learning_rate, 0.00001)
        self.assertEqual(rate_column.number_input.call_args.kwargs["min_value"], 0.00001)
        self.assertEqual(rate_column.number_input.call_args.kwargs["step"], 0.00001)
        self.assertEqual(rate_column.number_input.call_args.kwargs["format"], "%.5f")


if __name__ == "__main__":
    unittest.main()
