from datetime import datetime
import warnings
from typing import Tuple
from wazush.preproccess import load_data
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor, Ridge
from sklearn.metrics import f1_score


from IMLearn.base import BaseEstimator
import geopandas
import numpy as np
import pandas as pd
import pycountry






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


def pre_forxy(data: pd.DataFrame, labels) -> pd.DataFrame:
    data = data[["event_1_x", "event_1_y", "event_2_x", "event_2_y", "event_3_x", "event_3_y", "event_4_x"]]
    from scipy.spatial import distance
    # distance between two points
    data["dist_1_2"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x", "event_1_y"]], x[["event_2_x", "event_2_y"]]), axis=1)
    data["dist_1_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x", "event_1_y"]], x[["event_3_x", "event_3_y"]]), axis=1)
    data["dist_1_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x", "event_1_y"]], x[["event_4_x", "event_4_y"]]), axis=1)
    data["dist_2_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_x", "event_2_y"]], x[["event_3_x", "event_3_y"]]), axis=1)
    data["dist_2_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_x", "event_2_y"]], x[["event_4_x", "event_4_y"]]), axis=1)
    data["dist_3_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_3_x", "event_3_y"]], x[["event_4_x", "event_4_y"]]), axis=1)

class EstimatorXY:
    def _init_(self) -> BaseEstimator:
        self._classifier = RandomForestClassifier()

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')