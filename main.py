import numpy as np
# from utils import subtypes_cols, types_cols
from wazush.estimator import Estimator
from wazush.utils import types_cols, subtypes_cols
from wazush.preproccess import preprocess_first_task, load_data, split_data

if __name__ == '__main__':
    np.random.seed(0)

    df, labels = preprocess_first_task(load_data('waze_data.csv'))

    training_X, training_y, baseline_X, baseline_y, evaluation_X, evaluation_y, test_X, test_y = split_data(df, labels)
    baseline_y_classifier = baseline_y.drop(['x', 'y'], axis=1)
    classifier_labels = training_y.drop(['x', 'y'], axis=1)
    classifier_labels_type = training_y.drop(['x', 'y'] + subtypes_cols, axis=1)
    reg_labels = training_y.drop(types_cols + subtypes_cols, axis=1)

    model = Estimator()

    model.fit(training_X.to_numpy(), classifier_labels)

    pred = model.predict(training_X.to_numpy())
    print("The f1score is:", model.loss_classifier(training_X, classifier_labels))
    print("The f1score is:", model.loss_classifier(baseline_X, baseline_y_classifier))
    print("yay!!11")
