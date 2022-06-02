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


class Estimator:

    def __init__(self, estimators, depth):
        # Baseline
        # self._classifier = DecisionTreeClassifier()

        # f1 on train - 0.666, f1 on evaluation - 0.097
        # max evaluation f1score: 0.108, with max_depth=18
        # self._classifier = RandomForestClassifier(max_depth=estimators)

        # f1 on train - 0.666, f1 on evaluation - 0.099
        # max evaluation score: 0.103, with max_depth=11
        self._classifier = ExtraTreesClassifier(n_estimators=estimators,
        max_depth=depth)

        # f1 on train - 0.666, f1 on evaluation - 0.136 (1 neighbor)
        # self._classifier = KNeighborsClassifier(n_neighbors=estimators)

        # f1 on train - , f1 on evaluation -
        # self._classifier = BaggingClassifier(ExtraTreesClassifier(
        #     n_estimators=estimators))

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
