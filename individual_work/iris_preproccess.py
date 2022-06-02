from datetime import datetime
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
import pycountry
from scipy.stats import stats
from sklearn.neighbors import KDTree
import plotly.express as px
import plotly.graph_objects as go


def analyze_data(df: pd.DataFrame):
    df['pubDate'] = df['pubDate'].apply(lambda x: datetime.strptime(x, '%m/%d/%Y %H:%M:%S'))
    df['update_date'] = df['update_date'].apply(lambda d: datetime.utcfromtimestamp(d / 1000))
    df['day'] = df['update_date'].apply(lambda d: d.day)
    df['day_in_month'] = df['update_date'].apply(lambda d: d.days_in_month)
    df['day_of_week'] = df['update_date'].apply(lambda d: d.day_of_week)
    df['month'] = df['update_date'].apply(lambda d: d.month)
    df['year'] = df['update_date'].apply(lambda d: d.year)
    df['hour'] = df['update_date'].apply(lambda d: d.hour)
    df['minute'] = df['update_date'].apply(lambda d: d.minute)

    # calculate number of reports by day of the week
    hours = ["8:00-10:00", "12:00-14:00", "18:00-20:00"]
    my_df = pd.DataFrame()
    my_df['hours'] = hours
    my_df['number of events reported'] = [
        len(df[df['hour'] == 8]) + len(df[df['hour'] == 9]) + len(df[df['hour'] == 10]),
        len(df[df['hour'] == 12]) + len(df[df['hour'] == 13]) + len(df[df['hour'] == 14]),
        len(df[df['hour'] == 18]) + len(df[df['hour'] == 19]) + len(df[df['hour'] == 20])]
    fig = px.bar(my_df, x='hours', y='number of events reported',
                 title='The number of reports as a function of the hours',
                 )
    fig.write_image("C:/Users/iris/PycharmProjects/IML.HUJI/wazush/gr1.png")
    fig.write_html("C:/Users/iris/PycharmProjects/IML.HUJI/wazush/gr1.html")

    # plot x1 and x2
    df1 = df[['x', 'y', 'linqmap_city']]
    df1['linqmap_city'] = df1['linqmap_city'].fillna('other')
    fig1 = px.scatter(df1, x='x', y="y", color="linqmap_city",
                      title="Location of reports")
    fig1.write_image("C:/Users/iris/PycharmProjects/IML.HUJI/wazush/gr2.png")
    fig1.write_html("C:/Users/iris/PycharmProjects/IML.HUJI/wazush/gr2.html")

    # plot x1 and x2 only Tel Aviv
    df1 = df[['x', 'y', 'linqmap_city']]
    # df1['linqmap_city'] = df1['linqmap_city'].fillna('other')
    df1 = df1.loc[df1['linqmap_city'] == 'תל אביב - יפו']
    fig2 = px.scatter(df1, x='x', y="y", color="linqmap_city",
                      title="Location of reports")
    fig2.write_image("C:/Users/iris/PycharmProjects/IML.HUJI/wazush/gr3.png")
    fig2.write_html("C:/Users/iris/PycharmProjects/IML.HUJI/wazush/gr3.html")



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
    df = load_data('../waze_data.csv')
    analyze_data(df)
    print("done")
    # df, labels = preprocess_first_task(df)

    # training, baseline, evaluation, test = split_data(df)
