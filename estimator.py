from __future__ import annotations

from typing import NoReturn, Tuple

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
        self._classifier = MultiOutputClassifier(KNeighborsClassifier(225))
        # self._classifier = RandomForestClassifier()

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
