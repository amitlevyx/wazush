import pandas as pd
import numpy as np
from preproccess import preprocess_first_task, split_data
from sklearn.linear_model import LinearRegression


def task1_baseline_score(pred, real):
    return np.power(pred[0] - real[0], 2) + np.power(pred[1] - real[1], 2)


def task1_baseline(train_x, train_y):
    learner = LinearRegression()
    learner.fit(train_x, train_y)
    return learner


def task2_baseline(date, data):
    """
    Task 2 baseline
    """
    given_day = date.weekday()
    relevant_data = data[data['weekday'] == given_day]
    morning_data = relevant_data[relevant_data['morning_slot'] == 1]
    noon_data = relevant_data[relevant_data['noon_slot'] == 1]
    evening_data = relevant_data[relevant_data['evening_slot'] == 1]
    events = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM', 'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD']

    morning_pred = []
    noon_pred = []
    evening_pred = []
    for event in events:
        morning_pred.append((morning_data[event]).sum())
        noon_pred.append((noon_data[event]).sum())
        evening_pred.append((evening_data[event]).sum())
    df = pd.DataFrame()
    morning_pred, noon_pred, evening_pred = pd.Series(morning_pred), pd.Series(noon_pred), pd.Series(evening_pred)
    morning_pred.name, noon_pred.name, evening_pred.name = 'morning', 'noon', 'evening'
    df = df.append(morning_pred).append(noon_pred).append(evening_pred)

    return df
