from typing import Any, BinaryIO

import numpy as np
import polars as pl
from scipy.sparse import issparse


def to_dense_array(values: Any) -> np.ndarray:
    if issparse(values):
        return values.toarray()
    return np.asarray(values)


def read_tabular_data(data: BinaryIO, filename: str) -> pl.DataFrame:
    if filename.lower().endswith(".csv"):
        return pl.read_csv(data)
    if filename.lower().endswith(".xlsx"):
        return pl.read_excel(data)
    raise ValueError("Only CSV and XLSX files are supported.")
