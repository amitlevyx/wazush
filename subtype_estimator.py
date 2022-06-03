from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    BaggingClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class SubtypeEstimator:

    def __init__(self, estimators, depth):
        # # Baseline
        # self._classifier = DecisionTreeClassifier()

        # f1 on train - 0.571, f1 on evaluation - 0.059
        # max subtypes evaluation f1score: 0.0633, with max_depth=21
        self._classifier = RandomForestClassifier(n_estimators=estimators,
        max_depth=depth)

        # f1 on train - 0.571, f1 on evaluation - 0.057
        # max subtypes evaluation score: 0.0621, with max_depth=4
        # self._classifier = ExtraTreesClassifier(max_depth=estimators)

        # f1 on train - 0.571, f1 on evaluation - 0.088 (1 neighbor)
        # self._classifier = KNeighborsClassifier(n_neighbors=estimators)

        # f1 on train - , f1 on evaluation -
        # self._classifier = BaggingClassifier(ExtraTreesClassifier(
        #     n_estimators=estimators))

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
