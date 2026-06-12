import unittest

import numpy as np
import polars as pl
from sklearn.feature_selection import SelectKBest

from mlstudio.backend import FeatureSelectionConfig, get_transformed_feature_count
from mlstudio.backend.feature_selection import SCORE_FUNCTIONS, create_feature_selector


class FeatureSelectionTests(unittest.TestCase):
    def test_counts_features_created_by_preprocessing(self) -> None:
        preprocessing = pl.DataFrame(
            {
                "Variable": ["category", "rank", "amount", "enabled"],
                "Type": ["String", "String", "Numeric", "Boolean"],
                "Unique Count": [4, 12, 20, 2],
                "Preprocessing": [
                    "OneHotEncoder",
                    "OrdinalEncoder",
                    "StandardScaler",
                    "None",
                ],
            }
        )

        self.assertEqual(get_transformed_feature_count(preprocessing), 7)

    def test_all_score_functions_fit_select_k_best(self) -> None:
        random = np.random.default_rng(42)
        features = random.normal(size=(40, 5))
        target = 3 * features[:, 0] - features[:, 2] + random.normal(scale=0.1, size=40)

        for method in SCORE_FUNCTIONS:
            with self.subTest(method=method):
                selector = create_feature_selector(
                    FeatureSelectionConfig(method=method, count=2)
                )
                assert isinstance(selector, SelectKBest)
                transformed = selector.fit_transform(features, target)
                self.assertEqual(transformed.shape, (40, 2))


if __name__ == "__main__":
    unittest.main()
