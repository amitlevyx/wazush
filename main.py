import numpy as np
import pandas as pd
import baselines
# from estimator import Estimator
# from utils import *
from preproccess import preprocess_first_task, load_data, split_data, \
    preprocess_task2
from subtype_estimator import SubtypeEstimator
from estimator import Estimator_REG
from type_estimator import TypeEstimator


# def run_baseline(df, labels):
#     training_X, training_y, baseline_X, baseline_y, evaluation_X, \
#     evaluation_y, test_X, test_y = split_data(df, labels)
#
#     baseline_y_classifier = baseline_y.drop(['x', 'y'], axis=1)
#     classifier_labels = training_y.drop(['x', 'y'], axis=1)
#
#     # model = Estimator()
#     # model.fit_classifier(training_X.to_numpy(), classifier_labels.to_numpy())
#     # pred = model.predict_classifier(training_X.to_numpy())
#     # print("The training f1score is:", model.loss_classifier(training_X,
#     #                                                         classifier_labels))
#     # print("The baseline f1score is:", model.loss_classifier(baseline_X,
#     #                                                     baseline_y_classifier))
#
#     baseline_y_classifier_types = baseline_y.drop(['x', 'y'], axis=1)
#     baseline_y_classifier_types = baseline_y_classifier_types.drop(
#         subtypes_cols, axis=1)
#     classifier_type_labels = training_y.drop(['x', 'y'], axis=1)
#     classifier_type_labels = classifier_type_labels.drop(subtypes_cols, axis=1)
#
#     types_model = TypeEstimator()
#     types_model.fit_classifier(training_X.to_numpy(),
#                                classifier_type_labels.to_numpy())
#     types_pred = types_model.predict_classifier(training_X.to_numpy())
#     print("The types training f1score is:",
#           types_model.loss_classifier(training_X,
#                                       classifier_type_labels))
#     print("The types baseline f1score is:",
#           types_model.loss_classifier(baseline_X,
#                                       baseline_y_classifier_types))
#
#     baseline_y_classifier_subtypes = baseline_y.drop(['x', 'y'], axis=1)
#     baseline_y_classifier_subtypes = baseline_y_classifier_subtypes.drop(
#         types_cols, axis=1)
#     classifier_subtype_labels = training_y.drop(['x', 'y'], axis=1)
#     classifier_subtype_labels = classifier_subtype_labels.drop(types_cols,
#                                                                axis=1)
#
#     subtypes_model = SubtypeEstimator()
#     subtypes_model.fit_classifier(training_X.to_numpy(),
#                                   classifier_subtype_labels.to_numpy())
#     subtypes_pred = subtypes_model.predict_classifier(training_X.to_numpy())
#     print("The subtypes training f1score is:", subtypes_model.loss_classifier(
#         training_X, classifier_subtype_labels))
#     print("The subtypes baseline f1score is:", subtypes_model.loss_classifier(
#         baseline_X, baseline_y_classifier_subtypes))

def validation(df, labels):
    # types_cols = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM',
    #               'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD']
    # subtypes_cols = ['linqmap_subtype_ACCIDENT_MAJOR',
    #                  'linqmap_subtype_ACCIDENT_MINOR',
    #                  'linqmap_subtype_HAZARD_ON_ROAD',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_CAR_STOPPED',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_CONSTRUCTION',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_OBJECT',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_POT_HOLE',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
    #                  'linqmap_subtype_HAZARD_ON_SHOULDER_CAR_STOPPED',
    #                  'linqmap_subtype_JAM_HEAVY_TRAFFIC',
    #                  'linqmap_subtype_JAM_MODERATE_TRAFFIC',
    #                  'linqmap_subtype_JAM_STAND_STILL_TRAFFIC',
    #                  'linqmap_subtype_ROAD_CLOSED_CONSTRUCTION',
    #                  'linqmap_subtype_ROAD_CLOSED_EVENT']
    types_cols = ['linqmap_type']
    subtypes_cols = ['linqmap_subtype']

    training_X, training_y, baseline_X, baseline_y, evaluation_X, \
    evaluation_y, test_X, test_y = split_data(df, labels)

    training_X = pd.concat([training_X, baseline_X], axis=0)
    training_y = pd.concat([training_y, baseline_y], axis=0)

    # max_evaluation_scores = []
    max_types_evaluation_scores = []
    max_subtypes_evaluation_scores = []

    for est in range(20, 221, 20):
        # evaluation_scores = []
        types_evaluation_scores = []
        subtypes_evaluation_scores = []
        # for dep in range(5, 21):
        validation_y_classifier = evaluation_y.drop(['x', 'y'], axis=1)
        classifier_labels = training_y.drop(['x', 'y'], axis=1)

        # model = Estimator(est, dep)
        # model.fit_classifier(training_X.to_numpy(), classifier_labels.to_numpy())
        # pred = model.predict_classifier(
        #     evaluation_X.to_numpy())
        # evaluation_scores.append(model.loss_classifier(evaluation_X,
        #                                                validation_y_classifier))
        # print("The training f1score is:", model.loss_classifier(training_X,
        #                                                         classifier_labels))
        # print("The evaluation f1score is:", model.loss_classifier(evaluation_X,
        #                                                         validation_y_classifier))

        validation_y_classifier_types = validation_y_classifier.drop(
            subtypes_cols, axis=1)
        classifier_type_labels = classifier_labels.drop(subtypes_cols, axis=1)

        # types_model = TypeEstimator(est, dep)
        types_model = TypeEstimator(est)
        types_model.fit_classifier(training_X.to_numpy(),
                                   classifier_type_labels.to_numpy())
        # types_pred = types_model.predict_classifier(
        #     evaluation_X.to_numpy())
        types_evaluation_scores.append(types_model.loss_classifier(
            evaluation_X, validation_y_classifier_types))
        # print("The types training f1score is:",
        #       types_model.loss_classifier(training_X,
        #                                   classifier_type_labels))
        print("The types validation f1score is:",
              types_model.loss_classifier(evaluation_X,
                                          validation_y_classifier_types))

        validation_y_classifier_subtypes = validation_y_classifier.drop(
            types_cols, axis=1)
        classifier_subtype_labels = classifier_labels.drop(types_cols, axis=1)

        # subtypes_model = SubtypeEstimator(est, dep)
        subtypes_model = SubtypeEstimator(est)
        subtypes_model.fit_classifier(training_X.to_numpy(),
                                      classifier_subtype_labels.to_numpy())
        # subtypes_pred = subtypes_model.predict_classifier(
        #     evaluation_X.to_numpy())
        subtypes_evaluation_scores.append(subtypes_model.loss_classifier(
            evaluation_X, validation_y_classifier_subtypes))
        # print("The subtypes training f1score is:", subtypes_model.loss_classifier(
        #     training_X, classifier_subtype_labels))
        print("The subtypes validation f1score is:",
              subtypes_model.loss_classifier(
                  evaluation_X, validation_y_classifier_subtypes))
        print()

    # max_evaluation_scores.append(np.max(evaluation_scores))
    max_types_evaluation_scores.append(np.max(types_evaluation_scores))
    max_subtypes_evaluation_scores.append(np.max(
        subtypes_evaluation_scores))

    # print("num of estimators: ", est)
    # # print(f"max evaluation score: {np.max(evaluation_scores)}, "
    # #       f"with max depth of {np.argmax(evaluation_scores)}")
    print(f"max type evaluation score: {np.max(types_evaluation_scores)}, "
          f"with max depth of {np.argmax(types_evaluation_scores)}")
    print(f"max subtypes evaluation score: {np.max(subtypes_evaluation_scores)}, "
          f"with max depth of {np.argmax(subtypes_evaluation_scores)}")

    # print(f"max evaluation score: {np.max(max_evaluation_scores)}, "
    #       f"with {np.argmax(max_evaluation_scores)} estimator")
    # print(f"max type evaluation score: "
    #       f"{np.max(max_types_evaluation_scores)}, "
    #       f"with {np.argmax(max_types_evaluation_scores)}")
    # print(
    #     f"max subtypes evaluation score: {np.max(max_subtypes_evaluation_scores)}, "
    #     f"with {np.argmax(max_subtypes_evaluation_scores)}")


def extract_types_from_pred(types_pred):
    types = ['ACCIDENT', 'JAM', 'ROAD_CLOSED', 'WEATHERHAZARD']
    indexes = np.where(types_pred == 1)
    predictions = np.apply_along_axis(lambda row: types[row], 0, indexes)
    return predictions


def extract_subtypes_from_pred(subtypes_pred):
    types = ['ACCIDENT_MAJOR', 'ACCIDENT_MINOR', 'HAZARD_ON_ROAD',
             'HAZARD_ON_ROAD_CAR_STOPPED',
             'HAZARD_ON_ROAD_CONSTRUCTION',
             'HAZARD_ON_ROAD_OBJECT', 'HAZARD_ON_ROAD_POT_HOLE',
             'HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
             'HAZARD_ON_SHOULDER_CAR_STOPPED', 'JAM_HEAVY_TRAFFIC',
             'JAM_MODERATE_TRAFFIC', 'JAM_STAND_STILL_TRAFFIC',
             'ROAD_CLOSED_CONSTRUCTION', 'ROAD_CLOSED_EVENT']
    indexes = np.where(subtypes_pred == 1)
    predictions = np.apply_along_axis(lambda row: types[row], 0, indexes)
    return predictions


def run_task1(path: str, X):
    np.random.seed(0)
    types_cols = ['linqmap_type']
    subtypes_cols = ['linqmap_subtype']

    data, labels = preprocess_first_task(load_data(path))
    training_X, training_y, baseline_X, baseline_y, evaluation_X, \
    evaluation_y, test_X, test_y = split_data(data, labels)
    training_X = pd.concat([training_X, baseline_X, evaluation_X], axis=0)
    training_y = pd.concat([training_y, baseline_y, evaluation_y], axis=0)

    data = training_X
    labels = training_y

    classifier_labels = labels.drop(['x', 'y'], axis=1)
    type_model_labels = classifier_labels.drop(subtypes_cols, axis=1)
    subtype_model_labels = classifier_labels.drop(types_cols, axis=1)

    # Initialize and fit models to predict x, y, type and subtype of next
    # event for each sequence

    types_model = TypeEstimator(est=180)
    types_model.fit_classifier(data, type_model_labels)
    subtypes_model = SubtypeEstimator(est=20)
    subtypes_model.fit_classifier(data, subtype_model_labels)
    x_y_model = Estimator_REG()
    x_y_model.fit_reg(data, labels)

    # preprocess X

    types_pred = types_model.predict_classifier(X)
    print("types loss: ", types_model.loss_classifier(X, test_y['linqmap_type']))
    # types_pred = extract_types_from_pred(types_pred)
    subtypes_pred = subtypes_model.predict_classifier(X)
    print("subtypes loss: ", subtypes_model.loss_classifier(X, test_y['linqmap_subtype']))
    # subtypes_pred = extract_subtypes_from_pred(subtypes_pred)

    # predict x ,y
    x_pred, y_pred = x_y_model.predict_reg(X)
    print("xy loss: ", x_y_model.loss_reg(X, test_y.drop(['linqmap_type', 'linqmap_subtype'], axis=1)))

    # merge prediction
    prediction = np.concatenate([types_pred, subtypes_pred, x_pred.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
    return prediction


if __name__ == '__main__':
    np.random.seed(0)
    # types_cols = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM',
    #               'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD']
    # subtypes_cols = ['linqmap_subtype_ACCIDENT_MAJOR',
    #                  'linqmap_subtype_ACCIDENT_MINOR',
    #                  'linqmap_subtype_HAZARD_ON_ROAD',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_CAR_STOPPED',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_CONSTRUCTION',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_OBJECT',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_POT_HOLE',
    #                  'linqmap_subtype_HAZARD_ON_ROAD_TRAFFIC_LIGHT_FAULT',
    #                  'linqmap_subtype_HAZARD_ON_SHOULDER_CAR_STOPPED',
    #                  'linqmap_subtype_JAM_HEAVY_TRAFFIC',
    #                  'linqmap_subtype_JAM_MODERATE_TRAFFIC',
    #                  'linqmap_subtype_JAM_STAND_STILL_TRAFFIC',
    #                  'linqmap_subtype_ROAD_CLOSED_CONSTRUCTION',
    #                  'linqmap_subtype_ROAD_CLOSED_EVENT']
    # labels = labels.drop(columns=types_cols, axis=1)
    # labels = labels.drop(columns=subtypes_cols, axis=1)
    # task1_baseline = baselines.task1_baseline(training_X, training_y)
    # print(baselines.task1_baseline_score(task1_baseline.predict(baseline_X),
    # baseline_y.to_numpy()))
    # df = preprocess_task2(load_data('waze_data.csv'))
    # reg_labels = training_y.drop(types_and_subtypes_cols, axis=1)
    # validation(df, labels)

    df, labels = preprocess_first_task(load_data('waze_data.csv'))
    training_X, training_y, baseline_X, baseline_y, evaluation_X, \
    evaluation_y, test_X, test_y = split_data(df, labels)

    run_task1('waze_data.csv', test_X)

    # run_task1('waze_data.csv', test_X)
