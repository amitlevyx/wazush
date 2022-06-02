import numpy as np
import pandas as pd
import baselines
from estimator import Estimator
from preproccess import preprocess_first_task, load_data, split_data, \
    preprocess_task2
from subtype_estimator import SubtypeEstimator
from type_estimator import TypeEstimator

def run_baseline(df, labels):
    training_X, training_y, baseline_X, baseline_y, evaluation_X, \
    evaluation_y, test_X, test_y = split_data(df, labels)

    baseline_y_classifier = baseline_y.drop(['x', 'y'], axis=1)
    classifier_labels = training_y.drop(['x', 'y'], axis=1)

    model = Estimator()
    model.fit_classifier(training_X.to_numpy(), classifier_labels.to_numpy())
    pred = model.predict_classifier(training_X.to_numpy())
    print("The training f1score is:", model.loss_classifier(training_X,
                                                            classifier_labels))
    print("The baseline f1score is:", model.loss_classifier(baseline_X,
                                                        baseline_y_classifier))

    baseline_y_classifier_types = baseline_y.drop(['x', 'y'], axis=1)
    baseline_y_classifier_types = baseline_y_classifier_types.drop(
        subtypes_cols, axis=1)
    classifier_type_labels = training_y.drop(['x', 'y'], axis=1)
    classifier_type_labels = classifier_type_labels.drop(subtypes_cols, axis=1)

    types_model = TypeEstimator()
    types_model.fit_classifier(training_X.to_numpy(),
                               classifier_type_labels.to_numpy())
    types_pred = types_model.predict_classifier(training_X.to_numpy())
    print("The types training f1score is:",
          types_model.loss_classifier(training_X,
                                      classifier_type_labels))
    print("The types baseline f1score is:",
          types_model.loss_classifier(baseline_X,
                                      baseline_y_classifier_types))

    baseline_y_classifier_subtypes = baseline_y.drop(['x', 'y'], axis=1)
    baseline_y_classifier_subtypes = baseline_y_classifier_subtypes.drop(
        types_cols, axis=1)
    classifier_subtype_labels = training_y.drop(['x', 'y'], axis=1)
    classifier_subtype_labels = classifier_subtype_labels.drop(types_cols,
                                                               axis=1)

    subtypes_model = SubtypeEstimator()
    subtypes_model.fit_classifier(training_X.to_numpy(),
                                  classifier_subtype_labels.to_numpy())
    subtypes_pred = subtypes_model.predict_classifier(training_X.to_numpy())
    print("The subtypes training f1score is:", subtypes_model.loss_classifier(
        training_X, classifier_subtype_labels))
    print("The subtypes baseline f1score is:", subtypes_model.loss_classifier(
        baseline_X, baseline_y_classifier_subtypes))

def validation(df, labels):
    training_X, training_y, baseline_X, baseline_y, evaluation_X, \
    evaluation_y, test_X, test_y = split_data(df, labels)

    training_X = pd.concat([training_X, baseline_X], axis=0)
    training_y = pd.concat([training_y, baseline_y], axis=0)
    for k in range(40, 240, 20):
        print(k)
        validation_y_classifier = evaluation_y.drop(['x', 'y'], axis=1)
        classifier_labels = training_y.drop(['x', 'y'], axis=1)

        model = Estimator(k)
        model.fit_classifier(training_X.to_numpy(), classifier_labels.to_numpy())
        print("The training f1score is:", model.loss_classifier(training_X,
                                                                classifier_labels))
        print("The evaluation f1score is:", model.loss_classifier(evaluation_X,
                                                                validation_y_classifier))

        validation_y_classifier_types = validation_y_classifier.drop(
            subtypes_cols, axis=1)
        classifier_type_labels = classifier_labels.drop(subtypes_cols, axis=1)

        types_model = TypeEstimator(k)
        types_model.fit_classifier(training_X.to_numpy(),
                                   classifier_type_labels.to_numpy())
        print("The types training f1score is:",
              types_model.loss_classifier(training_X,
                                          classifier_type_labels))
        print("The types evaluation f1score is:",
              types_model.loss_classifier(evaluation_X,
                                          validation_y_classifier_types))

        validation_y_classifier_subtypes = validation_y_classifier.drop(
            types_cols, axis=1)
        classifier_subtype_labels = classifier_labels.drop(types_cols, axis=1)

        subtypes_model = SubtypeEstimator(k)
        subtypes_model.fit_classifier(training_X.to_numpy(),
                                      classifier_subtype_labels.to_numpy())
        print("The subtypes training f1score is:", subtypes_model.loss_classifier(
            training_X, classifier_subtype_labels))
        print("The subtypes baseline f1score is:", subtypes_model.loss_classifier(
            evaluation_X, validation_y_classifier_subtypes))
        print()

if __name__ == '__main__':
    np.random.seed(0)
    types_cols = ['linqmap_type_ACCIDENT', 'linqmap_type_JAM',
                  'linqmap_type_ROAD_CLOSED', 'linqmap_type_WEATHERHAZARD']
    subtypes_cols = ['linqmap_subtype_ACCIDENT_MAJOR',
                     'linqmap_subtype_ACCIDENT_MINOR',
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

    # labels = labels.drop(columns=types_cols, axis=1)
    # labels = labels.drop(columns=subtypes_cols, axis=1)
    # task1_baseline = baselines.task1_baseline(training_X, training_y)
    # print(baselines.task1_baseline_score(task1_baseline.predict(baseline_X),
    # baseline_y.to_numpy()))
    # df = preprocess_task2(load_data('waze_data.csv'))
    # reg_labels = training_y.drop(types_and_subtypes_cols, axis=1)
    # # to do - print(df.corr().abs().sort_values(ascending=False))

    df, labels = preprocess_first_task(load_data('waze_data.csv'))
    validation(df, labels)
    # print("yay!!11")
