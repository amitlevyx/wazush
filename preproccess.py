from datetime import datetime
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pycountry
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

    Returns:
        training 0.25
        baseline 0.125
        evaluation 0.125
        test 0.5
    """
    together = df.sample(frac=1).reset_index(drop=True)
    return together[:int(len(together) * 0.25)], \
           together[int(len(together) * 0.25):int(len(together) * 0.375)], \
           together[int(len(together) * 0.375):int(len(together) * 0.5)], \
           together[int(len(together) * 0.5):]


def preprocess(data: pd.DataFrame) -> pd.DataFrame:
    # todo add try except, pandas impute reliability
    data['pubDate'] = data['pubDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))
    data['update_date'] = data['update_date'].apply(lambda d: datetime.utcfromtimestamp(d / 1000))
    warnings.warn('update_date not found')
    data = pd.get_dummies(data, columns=['linqmap_type', 'linqmap_roadType', 'linqmap_subtype'])
    warnings.warn('linqmap_type not found')
    data = data.drop(
        columns=['linqmap_magvar', 'nComments', 'linqmap_reportMood', 'linqmap_nearby', 'linqmap_street',
                 'linqmap_expectedBeginDate', 'linqmap_reportDescription', 'linqmap_reportRating',
                 'linqmap_expectedEndDate'])
    print(data.columns)
    return data


if __name__ == '__main__':
    df = preprocess(load_data('waze_data.csv'))

    training, baseline, evaluation, test = split_data(df)
