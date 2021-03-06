from datetime import datetime
import warnings
from typing import Tuple
from wazush.preproccess import load_data

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import stats
from sklearn.neighbors import KDTree


def split_data(df: pd.DataFrame, labels) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
           data[int(len(together) * 0.25):int(len(together) * 0.375)], y[int(len(together) * 0.25):int(
        len(together) * 0.375)], \
           data[int(len(together) * 0.375):int(len(together) * 0.5)], y[int(len(together) * 0.375):int(
        len(together) * 0.5)], \
           data[int(len(together) * 0.5):], y[int(len(together) * 0.5):]


def preprocess_first_task(data: pd.DataFrame) -> pd.DataFrame:
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
    return data, labels



preprocess_first_task(load_data('waze_data.csv'))
