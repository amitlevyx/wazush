from wazush.estimator import Estimator
from wazush.preproccess import preprocess_first_task, load_data, split_data

if __name__ == '__main__':
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

    training_X, training_y, baseline_X, baseline_y, evaluation_X, evaluation_y, test_X, test_y = split_data(df, labels)
    classifier_labels = training_y.drop(['x', 'y'], axis=1)
    reg_labels = training_y.drop(types_and_subtypes_cols, axis=1)
    model = Estimator()

    model.fit_classifier(training_X.to_numpy(), classifier_labels.to_numpy())
    model.predict_classifier(training_X.to_numpy())

    print("yay!!11")
