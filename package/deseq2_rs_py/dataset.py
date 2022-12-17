import pandas as pd
from formulaic import Formula


class Dataset:
    """
    A Dataset object is a container for count data and column data. It is used to
    initialize a DESeq2 object.
    """

    def __init__(self, count_data: pd.DataFrame, col_data: pd.DataFrame, design: str):
        self.count_data = count_data
        self.col_data = col_data
        self.design = Formula(design).get_model_matrix()

        self._data_checks()

    def _data_checks(self):
        # check that the dataframe is square
        if not self.count_data.shape[0] == self.count_data.shape[1]:
            raise ValueError("Count data must be a matrix (square)")

        if not self.count_data.shape[1] == self.col_data.shape[0]:
            raise ValueError(
                "Count data must have the same number of columns as the column data has "
                "rows"
            )
        if self.count_data.isna().any().any():
            raise ValueError("Count data cannot contain NaN")
        if self.count_data.isnull().any().any():
            raise ValueError("Count data cannot contain null")
        if self.count_data[self.col_data < 0].any().any():
            raise ValueError("Count data cannot contain negative values")

    def _downcast_data(self):
        self.count_data = self.count_data.astype(int)
