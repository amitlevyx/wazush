import pandas as pd


from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor

from wazush.preproccess import preprocess_first_task, load_data, split_data

def pre_for_x(data: pd.DataFrame) -> pd.DataFrame:
    data = data[["event_1_x", "event_2_x", "event_3_x", "event_4_x"]]
    from scipy.spatial import distance
    # distance between two points
    data["dist_1_2"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x", 0]], x[["event_2_x", 0]]), axis=1)
    data["dist_1_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x", 0]], x[["event_3_x", 0]]), axis=1)
    data["dist_1_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_x", 0]], x[["event_4_x", 0]]), axis=1)
    data["dist_2_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_x", 0]], x[["event_3_x", 0]]), axis=1)
    data["dist_2_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_x", 0]], x[["event_4_x", 0]]), axis=1)
    data["dist_3_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_3_x", 0]], x[["event_4_x", 0]]), axis=1)

    # data.drop(["event_1_x", "event_2_x", "event_3_x", "event_4_x"], axis=1,
    #           inplace=True)
    return data


def pre_for_y(data: pd.DataFrame) -> pd.DataFrame:
    data = data[["event_1_y", "event_2_y", "event_3_y", "event_4_y"]]
    from scipy.spatial import distance
    # distance between two points
    data["dist_1_2"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_y",0]], x[["event_2_y"]]), axis=1)
    data["dist_1_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_y", 0]], x[["event_3_y", 0]]), axis=1)
    data["dist_1_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_1_y", 0]], x[["event_4_y", 0]]), axis=1)
    data["dist_2_3"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_y", 0]], x[["event_3_y", 0]]), axis=1)
    data["dist_2_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_2_y", 0]], x[["event_4_y", 0]]), axis=1)
    data["dist_3_4"] = data.apply(
        lambda x: distance.euclidean(x[["event_3_y", 0]], x[["event_4_y", 0]]), axis=1)

    # data.drop(["event_1_x", "event_2_x", "event_3_x", "event_4_x"], axis=1,
    #           inplace=True)
    return data

if __name__ == '__main__':
    df, labels = preprocess_first_task(load_data('waze_data.csv'))
    df = pre_for_x(df)
    training_X, training_y, baseline_X, baseline_y, evaluation_X, evaluation_y, test_X, test_y = split_data(df, labels)
    df_cor_x = pre_for_x(training_X)
    df_cor_y = pre_for_y(training_X)
    clf = MultiOutputRegressor(RandomForestRegressor(max_depth=4, random_state=0))
    clf.fit(df_cor_x, df_cor_y)
    print(clf.score(evaluation_X, evaluation_y))
    clf.predict(training_X)

