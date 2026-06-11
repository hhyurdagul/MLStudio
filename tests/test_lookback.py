import unittest

import numpy as np
from sklearn.linear_model import LinearRegression

from mlstudio.backend.lookback import AutoregressiveRegressor


class LookbackTests(unittest.TestCase):
    def test_fit_builds_consecutive_target_lags(self) -> None:
        estimator = AutoregressiveRegressor(LinearRegression(), lookback=3)
        features = np.arange(6, dtype=float).reshape(-1, 1)
        target = np.asarray([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])

        estimator.fit(features, target)

        np.testing.assert_array_equal(
            estimator.target_history_,
            [40.0, 50.0, 60.0],
        )
        self.assertEqual(
            estimator.get_feature_names_out(["feature"]).tolist(),
            ["feature", "target_lag_1", "target_lag_2", "target_lag_3"],
        )

    def test_prediction_is_recursive_and_resets_for_each_call(self) -> None:
        features = np.zeros((6, 1))
        target = np.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 13.0])
        estimator = AutoregressiveRegressor(LinearRegression(), lookback=2)
        estimator.fit(features, target)

        first = estimator.predict(np.zeros((3, 1)))
        second = estimator.predict(np.zeros((3, 1)))
        augmented = estimator.prediction_features(np.zeros((3, 1)))

        np.testing.assert_allclose(first, second)
        np.testing.assert_allclose(augmented[0, -2:], [13.0, 8.0])
        np.testing.assert_allclose(augmented[1, -2:], [first[0], 13.0])
        np.testing.assert_allclose(augmented[2, -2:], [first[1], first[0]])

    def test_scoring_uses_actual_fold_history(self) -> None:
        features = np.zeros((8, 1))
        target = np.asarray([1.0, 2.0, 3.0, 5.0, 8.0, 13.0, 21.0, 34.0])
        estimator = AutoregressiveRegressor(LinearRegression(), lookback=2)
        estimator.fit(features[:6], target[:6])

        predictions = estimator.predict_with_actual_history(
            features[6:],
            target[6:],
        )

        expected_rows = np.asarray(
            [
                [0.0, 13.0, 8.0],
                [0.0, 21.0, 13.0],
            ]
        )
        np.testing.assert_allclose(
            predictions,
            estimator.estimator_.predict(expected_rows),
        )

    def test_rejects_invalid_lookback(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one"):
            AutoregressiveRegressor(LinearRegression(), lookback=0).fit(
                np.zeros((2, 1)),
                np.zeros(2),
            )


if __name__ == "__main__":
    unittest.main()
