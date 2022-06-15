import pandas as pd

from dataset import Dataset


def deseq2(count_data: pd.DataFrame, col_data: pd.DataFrame, design_col: str):
    """
    DESeq2 analysis.
    """
    # Check data
    data = Dataset(count_data, col_data, design_col)
