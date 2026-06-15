import unittest

import numpy as np

from mlstudio.backend.elm import ELMClassifier, ELMRegressor


class ELMTests(unittest.TestCase):
    def test_regressor_fits_multi_output_targets(self) -> None:
        X = np.linspace(-1.0, 1.0, 40).reshape(-1, 1)
        y = np.column_stack((X[:, 0] ** 2, X[:, 0] ** 3))
        estimator = ELMRegressor(
            n_hidden=80,
            activation="tanh",
            alpha=1e-6,
            random_state=42,
        )

        predictions = estimator.fit(X, y).predict(X)

        self.assertEqual(predictions.shape, y.shape)
        self.assertLess(np.mean((predictions - y) ** 2), 1e-4)

    def test_regressor_is_reproducible(self) -> None:
        X = np.arange(12, dtype=float).reshape(-1, 1)
        y = X[:, 0] * 2.0
        first = ELMRegressor(random_state=42).fit(X, y)
        second = ELMRegressor(random_state=42).fit(X, y)

        np.testing.assert_allclose(first.predict(X), second.predict(X))

    def test_classifier_preserves_labels_and_returns_probabilities(self) -> None:
        X = np.array(
            [
                [-2.0, -1.0],
                [-1.5, -2.0],
                [1.5, 2.0],
                [2.0, 1.0],
            ]
        )
        y = np.array(["negative", "negative", "positive", "positive"])
        estimator = ELMClassifier(
            n_hidden=30,
            activation="tanh",
            alpha=1e-6,
            random_state=42,
        ).fit(X, y)

        np.testing.assert_array_equal(estimator.predict(X), y)
        probabilities = estimator.predict_proba(X)
        self.assertEqual(probabilities.shape, (4, 2))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)

    def test_invalid_hyperparameters_are_rejected(self) -> None:
        X = np.array([[0.0], [1.0]])
        y = np.array([0.0, 1.0])

        with self.assertRaisesRegex(ValueError, "n_hidden"):
            ELMRegressor(n_hidden=0).fit(X, y)
        with self.assertRaisesRegex(ValueError, "alpha"):
            ELMRegressor(alpha=-1.0).fit(X, y)


if __name__ == "__main__":
    unittest.main()
