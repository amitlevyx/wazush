from __future__ import annotations

from typing import NoReturn, Tuple
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, PoissonRegressor, TweedieRegressor, Ridge
from sklearn.metrics import f1_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def draw_diff(y_true, y_pred, x_true, x_pred):
    fig = make_subplots(rows=1, cols=2, subplot_titles=[f"Predicted",
                                                        f"True"])
    fig.add_trace(go.Scatter(x=x_pred, y=y_pred, mode="markers",
                             showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=x_true, y=y_true, mode="markers",
                             showlegend=False), row=1, col=2)
    fig.update_layout(title=f"The difference between the true x,y coordinates and the predicted coordinates",
                      xaxis_title="x", yaxis_title="y", title_x=0.5)
    fig.show()


def pre_label(type_ax, data: pd.DataFrame) -> pd.DataFrame:
    return data[type_ax]


def pre_for_x(data: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import distance
    new_data = data.copy()
    new_data["mean"] = new_data.apply(
        lambda x: (x["event_1_x"] + x["event_2_x"] + x["event_3_x"] + x["event_4_x"]) / 4, axis=1)
    new_data["var"] = new_data.apply(
        lambda x: (x["event_1_x"] - x["mean"]) ** 2 + (x["event_2_x"] - x["mean"]) ** 2 + (
                                            x["event_3_x"] - x["mean"]) ** 2 + (x["event_4_x"] - x["mean"]) ** 2, axis=1)
    new_data["dist_1_2_x"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_1_x"]], x[["event_2_x"]]), axis=1)
    new_data["dist_1_3_x"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_1_x"]], x[["event_3_x"]]), axis=1)
    new_data["dist_1_4_x"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_1_x"]], x[["event_4_x"]]), axis=1)
    new_data["dist_2_3_x"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_2_x"]], x[["event_3_x"]]), axis=1)
    new_data["dist_2_4_x"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_2_x"]], x[["event_4_x"]]), axis=1)
    new_data["dist_3_4_x"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_3_x"]], x[["event_4_x"]]), axis=1)
    new_data["mean_dist_x"] = new_data.apply(
        lambda x: (x["dist_1_2_x"] + x["dist_1_3_x"] + x["dist_1_4_x"] + x["dist_2_3_x"] + x["dist_2_4_x"] + x[
            "dist_3_4_x"]) / 6, axis=1)
    new_data["var_dist_x"] = new_data.apply(
        lambda x: (x["dist_1_2_x"] - x["mean_dist_x"]) ** 2 + (x["dist_1_3_x"] - x["mean_dist_x"]) ** 2 + (
                                                    x["dist_1_4_x"] - x["mean_dist_x"]) ** 2 + (x["dist_2_3_x"] - x[
            "mean_dist_x"]) ** 2 + (x["dist_2_4_x"] - x["mean_dist_x"]) ** 2 + (x["dist_3_4_x"] - x["mean_dist_x"]) ** 2, axis=1)
    new_data["centroid_dist_x"] = new_data.apply(
        lambda x: (x["dist_1_2_x"] + x["dist_1_3_x"] + x["dist_1_4_x"] + x["dist_2_3_x"] + x["dist_2_4_x"] + x[
            "dist_3_4_x"]) / 6, axis=1)
    new_data["linear"] = new_data.apply(
        lambda x: (x["event_1_x"] * x["event_2_x"] * x["event_3_x"] * x["event_4_x"]) / x["mean"], axis=1)
    new_data["quadratic"] = new_data.apply(
        lambda x: (x["event_1_x"] ** 2 * x["event_2_x"] ** 2 * x["event_3_x"] ** 2 * x["event_4_x"] ** 2) / x["mean"],
        axis=1)
    new_data["normalized_event_1_x"] = new_data.apply(
        lambda x: x["event_1_x"] / x["mean"], axis=1)
    new_data["normalized_event_2_x"] = new_data.apply(
        lambda x: x["event_2_x"] / x["mean"], axis=1)
    new_data["normalized_event_3_x"] = new_data.apply(
        lambda x: x["event_3_x"] / x["mean"], axis=1)
    new_data["normalized_event_4_x"] = new_data.apply(
        lambda x: x["event_4_x"] / x["mean"], axis=1)
    return new_data



def pre_for_y(data: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import distance
    new_data = data.copy()
    # distance between two points
    new_data["mean"] = new_data.apply(
        lambda x: (x["event_1_y"] + x["event_2_y"] + x["event_3_y"] + x["event_4_y"]) / 4, axis=1)
    new_data["var"] = new_data.apply(
        lambda x: (x["event_1_y"] - x["mean"]) ** 2 + (x["event_2_y"] - x["mean"]) ** 2 + (
                                x["event_3_y"] - x["mean"]) ** 2 + (x["event_4_y"] - x["mean"]) ** 2, axis=1)
    new_data["dist_1_2_y"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_1_y"]], x[["event_2_y"]]), axis=1)
    new_data["dist_1_3_y"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_1_y"]], x[["event_3_y"]]), axis=1)
    new_data["dist_1_4_y"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_1_y"]], x[["event_4_y"]]), axis=1)
    new_data["dist_2_3_y"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_2_y"]], x[["event_3_y"]]), axis=1)
    new_data["dist_2_4_y"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_2_y"]], x[["event_4_y"]]), axis=1)
    new_data["dist_3_4_y"] = new_data.apply(
        lambda x: distance.euclidean(x[["event_3_y"]], x[["event_4_y"]]), axis=1)
    new_data["mean_dist"] = new_data.apply(
        lambda x: (x["dist_1_2_y"] + x["dist_1_3_y"] + x["dist_1_4_y"] + x["dist_2_3_y"] + x["dist_2_4_y"] + x[
            "dist_3_4_y"]) / 6, axis=1)
    new_data["var_dist"] = new_data.apply(
        lambda x: (x["dist_1_2_y"] - x["mean_dist"]) ** 2 + (x["dist_1_3_y"] - x["mean_dist"]) ** 2 + (
                                                                x["dist_1_4_y"] - x["mean_dist"]) ** 2 + (x["dist_2_3_y"] - x[
            "mean_dist"]) ** 2 + (x["dist_2_4_y"] - x["mean_dist"]) ** 2 + (x["dist_3_4_y"] - x["mean_dist"]) ** 2, axis=1)
    new_data["normalized_linear"] = new_data.apply(
        lambda x: (x["event_1_y"] * x["event_2_y"] * x["event_3_y"] * x["event_4_y"]) / x["mean"] / x["var"],
        axis=1)
    new_data["normalized_linear_dist"] = new_data.apply(
        lambda x: (x["event_1_y"] * x["event_2_y"] * x["event_3_y"] * x["event_4_y"]) / x["mean_dist"] / x["var_dist"],
        axis=1)
    new_data["normalized_event_1_y"] = new_data.apply(
        lambda x: x["event_1_y"] / x["mean"], axis=1)
    new_data["normalized_event_2_y"] = new_data.apply(
        lambda x: x["event_2_y"] / x["mean"], axis=1)
    new_data["normalized_event_3_y"] = new_data.apply(
        lambda x: x["event_3_y"] / x["mean"], axis=1)
    new_data["normalized_event_4_y"] = new_data.apply(
        lambda x: x["event_4_y"] / x["mean"], axis=1)
    new_data["quadratic"] = new_data.apply(
        lambda x: (x["event_1_y"] * x["event_2_y"] * x["event_3_y"] * x["event_4_y"]) / x["mean"] ** 2, axis=1)
    new_data["quadratic_dist"] = new_data.apply(
        lambda x: (x["event_1_y"] * x["event_2_y"] * x["event_3_y"] * x["event_4_y"]) / x["mean_dist"] ** 2, axis=1)
    new_data["cubic"] = new_data.apply(
        lambda x: (x["event_1_y"] * x["event_2_y"] * x["event_3_y"] * x["event_4_y"]) / x["mean"] ** 3, axis=1)

    return new_data


def score_func(y_true, y_pred, x_true, x_pred):
    df = pd.DataFrame({"x_pred": x_pred, "y_pred": y_pred, "x_true": x_true, "y_true": y_true})
    df["score"] = df.apply(lambda x: ((x["x_true"] - x["x_pred"]) ** 2) + ((x["y_true"] - x["y_pred"]) ** 2), axis=1)
    return df["score"].mean()


class Estimator_REG:
    def __init__(self):
        self.estimator_x = RandomForestRegressor(n_estimators=1000, random_state=0, criterion='squared_error')
        self.estimator_y = RandomForestRegressor(n_estimators=1000, random_state=0, criterion='squared_error')
        # self._classifier = RandomForestClassifier()

    def fit_reg(self, X, y):
        self.estimator_x.fit(pre_for_x(X), pre_label("x", y))
        self.estimator_y.fit(pre_for_y(X), pre_label("y", y))

    def predict_reg(self, X):
        return self.estimator_x.predict(pre_for_x(X)), self.estimator_y.predict(pre_for_y(X))

    def loss_reg(self, X, y):
        x_pred, y_pred = self.predict_reg(X)
        return score_func(pre_label("y", y), y_pred, pre_label("x", y), x_pred)
    def draw_diff(self, X ,y):
        x_pred, y_pred = self.predict_reg(X)
        draw_diff(pre_label("y", y), y_pred, pre_label("x", y), x_pred)



