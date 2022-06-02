from datetime import datetime
import warnings
from typing import Tuple, Any

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


def split_data(df: pd.DataFrame, labels) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame,
                                                  pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    y_cols = labels.columns
    together = pd.concat([df, labels], axis=1)
    together = together.sample(frac=1).reset_index(drop=True)
    data, y = together.drop(list(y_cols), axis=1), together[y_cols]
    return data[:int(len(together) * 0.25)], y[:int(len(together) * 0.25)], \
           data[int(len(together) * 0.25):int(len(together) * 0.375)], y[int(len(together) * 0.25):int(
        len(together) * 0.375)], \
           data[int(len(together) * 0.375):int(len(together) * 0.5)], y[int(len(together) * 0.375):int(
        len(together) * 0.5)], \
           data[int(len(together) * 0.5):], y[int(len(together) * 0.5):]


def preprocess_first_task(data: pd.DataFrame) -> Tuple[Any, Any]:
    # todo add try except, pandas impute reliability
    data = data.loc[data['linqmap_city'] == 'תל אביב - יפו']
    linqmap_reliability_mean = data['linqmap_reliability'].mean()
    data['linqmap_reliability'].fillna(value=linqmap_reliability_mean, inplace=True)
    data['pubDate'] = data['pubDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))
    data['update_date'] = data['update_date'].apply(lambda d: datetime.utcfromtimestamp(d / 1000))
    data.sort_values(by='update_date', ascending=True)
    data = data.reset_index(drop=True)
    data = pd.get_dummies(data, columns=['linqmap_type', 'linqmap_roadType', 'linqmap_subtype'])
    data['day'] = data['update_date'].apply(lambda d: d.day)
    data['day_in_month'] = data['update_date'].apply(lambda d: d.days_in_month)
    data['day_of_week'] =data['update_date'].apply(lambda d: d.day_of_week)
    data['month'] = data['update_date'].apply(lambda d: d.month)
    data['year'] = data['update_date'].apply(lambda d: d.year)
    data['hour'] = data['update_date'].apply(lambda d: d.hour)
    data['minute'] = data['update_date'].apply(lambda d: d.minute)
    data = data.drop(
        columns=['linqmap_magvar', 'pubDate', 'update_date', 'nComments', 'linqmap_reportMood', 'linqmap_nearby',
                 'linqmap_street',
                 'linqmap_expectedBeginDate', 'linqmap_reportDescription', 'linqmap_reportRating',
                 'linqmap_expectedEndDate', 'linqmap_city'])
    # split to four events and fifth one

    data['number'] = np.ceil(data.index / 5)
    labels = data[data.index % 5 == 0]
    labels = labels[1:].reset_index(drop=True)
    data1, data2, data3, data4 = data[data.index % 5 == 1], data[data.index % 5 == 2], data[data.index % 5 == 3], data[
        data.index % 5 == 1]
    dfs = [data1, data2, data3, data4]
    for i in range(len(dfs)):
        dfs[i] = dfs[i].add_prefix('event_' + str(i + 1) + '_')
        dfs[i]['number'] = dfs[i]['event_' + str(i + 1) + '_number']
        dfs[i] = dfs[i].drop(columns=['event_' + str(i + 1) + '_number'])
    data = dfs[0].merge(dfs[1], on='number')
    data = data.merge(dfs[2], on='number')
    data = data.merge(dfs[3], on='number')
    data = data.drop(columns=['number'])
    labels = labels.drop(columns=['number'])
    labels = labels.drop(['OBJECTID', 'pubDate', 'linqmap_reliability', 'update_date', 'linqmap_roadType_1',
                          'linqmap_roadType_2', 'linqmap_roadType_4', 'linqmap_roadType_16',
                          'linqmap_roadType_17', 'linqmap_roadType_20', 'linqmap_roadType_22'], axis=1)
    return data, labels
