import random

import pandas as pd
from sklearn.model_selection import GridSearchCV

from baselines import *
from sklearn.metrics import mean_squared_error
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

from wazush.preproccess import preprocess_first_task, load_data, split_data


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


if __name__ == '__main__':
    np.random.seed(0)
    # np.random.seed(0)
    df, labels = preprocess_first_task(load_data('waze_data.csv'))
    training_X, training_y, baseline_X, baseline_y, evaluation_X, evaluation_y, test_X, test_y = split_data(df, labels)
    " combine training and baseline data "
    training_X = pd.concat([training_X,baseline_X], axis=0)
    training_y = pd.concat([training_y,baseline_y], axis=0)
    x_true = pre_label("x", training_y)
    y_true = pre_label("y", training_y)
    x_base = pre_for_x(baseline_X)
    y_base = pre_for_y(baseline_X)

    estimator_x_f = RandomForestRegressor(n_estimators=1000, random_state=18, criterion='squared_error')
    estimator_x_f.fit(pre_for_x(training_X), x_true)

    estimator_y_f = RandomForestRegressor(n_estimators=1000, random_state=18, criterion='squared_error')
    estimator_y_f.fit(pre_for_y(training_X), y_true)


    print("Random Forest")
    x_pred_1 = estimator_x_f.predict(pre_for_x(evaluation_X))
    y_pred_1 = estimator_y_f.predict(pre_for_y(evaluation_X))
    print(mean_squared_error(pre_label("x",evaluation_y), x_pred_1))
    print(mean_squared_error(pre_label("y",evaluation_y), y_pred_1))
    print(score_func(pre_label("y",evaluation_y), y_pred_1, pre_label("x",evaluation_y), x_pred_1))

    est = Estimator_REG()
    est.fit_reg(training_X, training_y)
    x_pred_2, y_pred_2 = est.predict_reg(evaluation_X)
    print(score_func(pre_label("y",evaluation_y), y_pred_2, pre_label("x",evaluation_y), x_pred_2))


    # print("Process")
    # x_pred = estimator_x_l.predict(pre_for_x(baseline_X))
    # y_pred = estimator_y_l.predict(pre_for_y(baseline_X))
    # print(mean_squared_error(pre_label("x",baseline_y), x_pred))
    # print(mean_squared_error(pre_label("y",baseline_y), y_pred))
    # print(score_func(pre_label("y",baseline_y), y_pred, pre_label("x",baseline_y), x_pred))
    #
    # print(mean_squared_error(pre_label("x", baseline_y), x_pred))
    # print(mean_squared_error(pre_label("y", baseline_y), y_pred))
    # print(score_func(pre_label("y", baseline_y), y_pred, pre_label("x", baseline_y), x_pred))



















# estimator_x_l = LinearRegression()
# together_X = np.column_stack((pre_for_x(training_X), pre_for_y(training_X)))
# together_y = np.column_stack((x_true, y_true))
# estimator_x_l.fit(together_X, together_y)
#
# together_X_base = np.column_stack((pre_for_x(baseline_X), pre_for_y(baseline_X)))
# together_true = baseline_y[["x", "y"]]
# pred = estimator_x_l.predict(together_X_base)
# print("Linear")
# print(mean_squared_error(together_true["x"], pred[:, 0]))
# print(mean_squared_error(together_true["y"], pred[:, 1]))
# print(mean_squared_error(together_true, pred))



