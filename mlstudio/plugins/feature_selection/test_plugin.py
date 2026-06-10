import unittest

import numpy as np
from sklearn.feature_selection import SelectKBest

from . import SCORE_FUNCTIONS, create_feature_selection_step


class FeatureSelectionPluginTests(unittest.TestCase):
    def test_all_score_functions_fit_select_k_best(self) -> None:
        random = np.random.default_rng(42)
        features = random.normal(size=(40, 5))
        target = (
            3 * features[:, 0]
            - features[:, 2]
            + random.normal(scale=0.1, size=40)
        )

        for method in SCORE_FUNCTIONS:
            with self.subTest(method=method):
                step = create_feature_selection_step(method, 2)
                selector = step.transformer
                assert isinstance(selector, SelectKBest)
                transformed = selector.fit_transform(features, target)
                self.assertEqual(transformed.shape, (40, 2))


if __name__ == "__main__":
    unittest.main()
