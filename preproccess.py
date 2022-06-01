import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pycountry

from IMLearn.base import BaseEstimator
from scipy.stats import stats



def load_data(path: str) -> pd.DataFrame:
    """
    Loads the data from the given path.

    Args:
        path: The path to the data.

    Returns:
        The data as a pandas dataframe.
    """
    # remove outliers
    return pd.read_csv(path)

def split_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits the data into training and test data.

    Args:
        df: The dataframe to split.
        test_size: The size of the test data.

    Returns:
        training 0.25
        base_estimator 0.125
        evaluation 0.125
        test 0.5
    """
    together = df.sample(frac=1).reset_index(drop=True)
    return together[:int(len(together)*0.25)], \
           together[int(len(together)*0.25):int(len(together)*0.5)], \
           together[int(len(together)*0.5):int(len(together)*0.75)], \
           together[int(len(together)*0.75):]


def preproccess(df: pd.DataFrame) -> pd.DataFrame:
    pass



