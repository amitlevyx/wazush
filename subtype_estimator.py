from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    BaggingClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class SubtypeEstimator:

    def __init__(self, estimators):
        # # Baseline
        # self._classifier = DecisionTreeClassifier()

        # f1 on train - 0.571, f1 on evaluation - 0.059
        # self._classifier = RandomForestClassifier(n_estimators=estimators)

        # f1 on train - 0.571, f1 on evaluation - 0.057
        # self._classifier = ExtraTreesClassifier()

        # f1 on train - 0.571, f1 on evaluation - 0.088 (1 neighbor)
        # self._classifier = KNeighborsClassifier(n_neighbors=estimators)

        # f1 on train - , f1 on evaluation -
        self._classifier = BaggingClassifier(ExtraTreesClassifier(
            n_estimators=estimators))

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
