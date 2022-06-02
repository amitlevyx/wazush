from __future__ import annotations

from typing import NoReturn, Tuple

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor, Ridge
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeClassifier

from IMLearn.base import BaseEstimator



def fit(X, y):
