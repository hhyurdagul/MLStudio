from typing import BinaryIO

import polars as pl


def read_tabular_data(data: BinaryIO, filename: str) -> pl.DataFrame:
    if filename.lower().endswith(".csv"):
        return pl.read_csv(data)
    if filename.lower().endswith(".xlsx"):
        return pl.read_excel(data)
    raise ValueError("Only CSV and XLSX files are supported.")
