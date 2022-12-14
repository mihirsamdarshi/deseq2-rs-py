from typing import Optional, List

import pandas as pd


class Dataset:
    """
    A Dataset object is a container for count data and column data. It is used to
    initialize a DESeq2 object.
    """
    def __init__(self, count_data: pd.DataFrame, col_data: pd.DataFrame, design_col: str):
        self.count_data = count_data
        self.col_data = col_data
        self.design_col = design_col

        self._data_checks()

    def _data_checks(self):
        if not self.count_data.shape[1] == self.col_data.shape[0]:
            raise ValueError(
                "Count data must have the same number of columns as the column data has rows"
            )
        if self.col_data.isna().any().any():
            raise ValueError("Column data cannot contain NaN")
        if self.col_data.isnull().any().any():
            raise ValueError("Column data cannot contain null")
        if self.col_data[self.col_data < 0].any().any():
            raise ValueError("Column data cannot contain negative values")

        if self.design_col not in self.col_data.columns:
            raise ValueError("Design column must be in column data")

    def _downcast_data(self):
        self.count_data = self.count_data.astype(int)
