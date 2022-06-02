from __future__ import annotations

from typing import NoReturn, Tuple
from wazush.utils import types_cols, subtypes_cols
import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor, Ridge
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from IMLearn.base import BaseEstimator


class Estimator:
    def __init__(self) -> BaseEstimator:
        # self._classifier = MultiOutputClassifier(KNeighborsClassifier(1))
        self.type_model_classifier = MultiOutputClassifier(KNeighborsClassifier(1))
        self.jam_classifier = MultiOutputClassifier(KNeighborsClassifier(1))
        self.accident_classifier = MultiOutputClassifier(KNeighborsClassifier(1))
        self.road_classifier = MultiOutputClassifier(KNeighborsClassifier(1))
        self.weather_classifier = MultiOutputClassifier(KNeighborsClassifier(1))
        # self._classifier = RandomForestClassifier()

    def fit(self, X, y):
        type_y = y[types_cols]
        subtype_y = y[subtypes_cols]
        self.type_model_classifier.fit(X, type_y.to_numpy())
        self.jam_classifier.fit(X, subtype_y.to_numpy())
        self.accident_classifier.fit(X, subtype_y.to_numpy())
        self.road_classifier.fit(X, subtype_y.to_numpy())
        self.weather_classifier.fit(X, subtype_y.to_numpy())

    def predict(self, X):
        type_pred = self.predict_type(X)
        type_pred_string = self.get_pred_type(type_pred)
        pred_sub = self.predict_subtype(X, type_pred_string)
        return np.concatenate(type_pred, pred_sub)

    def get_pred_type(self, prediction):
        return ""

    def predict_type(self, X):
        return self.type_model_classifier.predict(X)

    def predict_subtype(self, X, event_type: str):
        pred = None
        if event_type == "JAM":
            pred = self.jam_classifier.predict(X)
        elif event_type == "ACCIDENT":
            pred = self.accident_classifier.predict(X)
        elif event_type == "ROAD_CLOSED":
            pred = self.road_classifier.predict(X)
        elif event_type == "WEATHERHAZARD":
            pred = self.weather_classifier.predict(X)
        return pred

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
