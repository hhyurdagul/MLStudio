import unittest

import polars as pl

from mlstudio.backend.preprocessing import (
    create_preprocessing_transformer,
    get_preprocessing_data,
)


class PreprocessingTests(unittest.TestCase):
    def test_mixed_features_are_described_and_transformed(self) -> None:
        df = pl.DataFrame(
            {
                "number": [1, 2, 3],
                "category": ["a", "b", "a"],
                "flag": [True, False, True],
            }
        )

        preprocessing = get_preprocessing_data(df)
        transformer = create_preprocessing_transformer(preprocessing)
        transformed = transformer.fit_transform(df)

        self.assertEqual(set(preprocessing["Variable"]), set(df.columns))
        self.assertEqual(transformed.shape, (3, 4))

    def test_unsupported_feature_type_is_rejected(self) -> None:
        df = pl.DataFrame(
            {"date": [pl.date(2026, 1, 1), pl.date(2026, 1, 2)]}
        )

        with self.assertRaisesRegex(ValueError, "Unsupported feature types"):
            get_preprocessing_data(df)


if __name__ == "__main__":
    unittest.main()
