from datetime import datetime
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import stats
from sklearn.neighbors import KDTree


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


def preprocess_first_task(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # todo add try except, pandas impute reliability
    data = data.loc[data['linqmap_city'] == 'תל אביב - יפו']
    linqmap_reliability_mean = data['linqmap_reliability'].mean()
    data['linqmap_reliability'].fillna(value=linqmap_reliability_mean, inplace=True)
    data['pubDate'] = data['pubDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))
    data['update_date'] = data['update_date'].apply(lambda d: datetime.utcfromtimestamp(d / 1000))
    data.sort_values(by='update_date', ascending=True)
    data = data.reset_index(drop=True)
    data = pd.get_dummies(data, columns=['linqmap_type', 'linqmap_roadType', 'linqmap_subtype'])
    data = data.drop(
        columns=['linqmap_magvar', 'nComments', 'linqmap_reportMood', 'linqmap_nearby', 'linqmap_street',
                 'linqmap_expectedBeginDate', 'linqmap_reportDescription', 'linqmap_reportRating',
                 'linqmap_expectedEndDate'])
    # split to four events and fifth one

    labels = data[data.index % 5 == 0]
    data = data[data.index % 5 != 0]
    data = df.groupby(np.repeat(np.arange(len(df)), 4)[:len(df)]).agg(' '.join)
    return data, labels


if __name__ == '__main__':
    df, labels = preprocess_first_task(load_data('../waze_data.csv'))

    training, baseline, evaluation, test = split_data(df)
