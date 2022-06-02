import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import baselines
from estimator import Estimator
from preproccess import preprocess_first_task, load_data, split_data, preprocess_task2


# def plot_graph(data: pd.DataFrame):
#
#     # plot number of events per day
#
#     # groupby_event = data.groupby('Country', as_index=False).apply(calc_loss)
#     # figIsraelCountries = px.bar(groupby_country, x='Country', y='Loss',
#     #                             title="Countries loss over model fitted for Israel.")
#     # fig = go.Figure([go.Bar(x=data['day_of_week'], y=data.groupby("day_of_week"), name='accident'),
#     #                  go.Bar(x=data['day_of_week'], y=data[data['linqmap_type_JAM'] == 1].shape[0], name='jam'),
#     #                  go.Bar(x=data['day_of_week'], y=data[data['linqmap_type_ROAD_CLOSED'] == 1].shape[0], name='road closed'),
#     #                  go.Bar(x=data['day_of_week'], y=data[data['linqmap_type_WEATHERHAZARD'] == 1].shape[0], name='weather')])
#     # fig.show()
#     # data = data.groupby("day_of_week", as_index=False).apply(lambda x: )
#     # fig = px.bar(data, x="1", y=["linqmap_type_ROAD_CLOSED", "linqmap_type_ACCIDENT", "linqmap_type_JAM",
#     #                                        "linqmap_type_WEATHERHAZARD"])
#     # fig.show()
#

if __name__ == '__main__':
    np.random.seed(0)
    types_and_subtypes_cols = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM',
                               'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD',
                               'linqmap_subtype_ACCIDENT_MAJOR', 'linqmap_subtype_ACCIDENT_MINOR',
                               'linqmap_subtype_HAZARD_ON_ROAD',
                               'linqmap_subtype_HAZARD_ON_ROAD_CAR_STOPPED',
                               'linqmap_subtype_HAZARD_ON_ROAD_CONSTRUCTION',
                               'linqmap_subtype_HAZARD_ON_ROAD_OBJECT',
                               'linqmap_subtype_HAZARD_ON_ROAD_POT_HOLE',
                               'linqmap_subtype_HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
                               'linqmap_subtype_HAZARD_ON_SHOULDER_CAR_STOPPED',
                               'linqmap_subtype_JAM_HEAVY_TRAFFIC',
                               'linqmap_subtype_JAM_MODERATE_TRAFFIC',
                               'linqmap_subtype_JAM_STAND_STILL_TRAFFIC',
                               'linqmap_subtype_ROAD_CLOSED_CONSTRUCTION',
                               'linqmap_subtype_ROAD_CLOSED_EVENT']

    df, labels = preprocess_first_task(load_data('waze_data.csv'))
    # df = preprocess_task2(load_data('waze_data.csv'))
    # plot_graph(df)
    labels = labels.drop(columns=types_and_subtypes_cols, axis=1)
    training_X, training_y, baseline_X, baseline_y, evaluation_X, evaluation_y, test_X, test_y = split_data(df, labels)
    print(df.corrwith(labels).sort_values(ascending=False, key=abs).to_string())
    task1_baseline = baselines.task1_baseline(training_X, training_y)
    print(baselines.task1_baseline_score(task1_baseline.predict(baseline_X), baseline_y.to_numpy()))
    # df = preprocess_task2(load_data('waze_data.csv'))
    #
    # # baseline_y_classifier = baseline_y.drop(['x', 'y'], axis=1)
    # # classifier_labels = training_y.drop(['x', 'y'], axis=1)
    # # reg_labels = training_y.drop(types_and_subtypes_cols, axis=1)
    # # # todo: print(df.corr().abs().sort_values(ascending=False))
    #
    # model = Estimator()
    #
    # model.fit_classifier(training_X.to_numpy(), classifier_labels.to_numpy())
    # pred = model.predict_classifier(training_X.to_numpy())
    # print("The f1score is:", model.loss_classifier(training_X, classifier_labels))
    # print("The f1score is:", model.loss_classifier(baseline_X, baseline_y_classifier))
    # print("yay!!11")
