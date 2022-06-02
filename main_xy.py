import pandas as pd


from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from wazush.preproccess import preprocess_first_task, load_data, split_data

def pre_label(type_ax, data: pd.DataFrame) -> pd.DataFrame:
     return data[type_ax]


def pre_for_x(data: pd.DataFrame) -> pd.DataFrame:
    data = data[["event_1_x", "event_2_x", "event_3_x", "event_4_x"]]
    from scipy.spatial import distance
    # distance between two points
    data["dist_1_2"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x"]], x[["event_2_x"]]), axis=1)
    data["dist_1_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x"]], x[["event_3_x"]]), axis=1)
    data["dist_1_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x"]], x[["event_4_x"]]), axis=1)
    data["dist_2_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_x"]], x[["event_3_x"]]), axis=1)
    data["dist_2_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_x"]], x[["event_4_x"]]), axis=1)
    data["dist_3_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_3_x"]], x[["event_4_x"]]), axis=1)

    # data.drop(["event_1_x", "event_2_x", "event_3_x", "event_4_x"], axis=1,
    #           inplace=True)
    return data


def pre_for_y(data: pd.DataFrame) -> pd.DataFrame:
    from scipy.spatial import distance
    # distance between two points
    data["dist_1_2_y"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_y"]], x[["event_2_y"]]), axis=1)
    data["dist_1_3_y"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_y"]], x[["event_3_y"]]), axis=1)
    data["dist_1_4_y"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_y"]], x[["event_4_y"]]), axis=1)
    data["dist_2_3_y"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_y"]], x[["event_3_y"]]), axis=1)
    data["dist_2_4_y"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_y"]], x[["event_4_y"]]), axis=1)
    data["dist_3_4_y"] = data.apply(
        lambda x: distance.euclidean(x[["event_3_y"]], x[["event_4_y"]]), axis=1)

    # data.drop(["event_1_x", "event_2_x", "event_3_x", "event_4_x"], axis=1,
    #           inplace=True)
    return data

if __name__ == '__main__':
    df, labels = preprocess_first_task(load_data('waze_data.csv'))
    training_X, training_y, baseline_X, baseline_y, evaluation_X, evaluation_y, test_X, test_y = split_data(df, labels)
    # kernel = 1 * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    estimator_x = RandomForestRegressor(n_estimators = 100, random_state = 0)
    estimator_x.fit(pre_for_x(training_X), pre_label("x", training_y))
    print(estimator_x.score(baseline_X, pre_label("x", baseline_y)))
