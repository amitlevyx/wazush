from datetime import datetime
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import stats
from sklearn.neighbors import KDTree

def split_data(df: pd.DataFrame, labels) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and test data.

    Args:
        df: The dataframe to split.

    Returns:
        training features and labels 0.25
        baseline features nd labels 0.125
        evaluation 0.125
        test 0.5
    """
    together = df.join(labels)
    together = together.sample(frac=1).reset_index(drop=True)
    data, y = together.iloc[:, :-1], together.iloc[:, -1]
    return data[:int(len(together) * 0.25)], y[:int(len(together) * 0.25)], \
           data[int(len(together) * 0.25):int(len(together) * 0.375)], y[int(len(together) * 0.25):int(len(together) * 0.375)],\
           data[int(len(together) * 0.375):int(len(together) * 0.5)], y[int(len(together) * 0.375):int(len(together) * 0.5)], \
           data[int(len(together) * 0.5):], y[int(len(together) * 0.5):]