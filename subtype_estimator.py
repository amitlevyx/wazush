from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class SubtypeEstimator:

    def __init__(self, est):
        # # Baseline
        # self._classifier = DecisionTreeClassifier()

        # f1 on evaluation - 0.059
        # max subtypes evaluation f1score: 0.0633, with max_depth=21
        # self._classifier = RandomForestClassifier(n_estimators=estimators,
        # max_depth=depth)
        # f1 on evaluation - 0.143
        # self._classifier = RandomForestClassifier()

        # f1 on evaluation - 0.143
        # self._classifier = ExtraTreesClassifier(n_estimators=est)

        # f1 on evaluation - 0.088 (1 neighbor)
        # self._classifier = KNeighborsClassifier(n_neighbors=estimators)

        # f1 on evaluation - 0.145
        # 20 est - 0.176
        self._classifier = GradientBoostingClassifier(n_estimators=est)


        # f1 on evaluation - 0.142
        # self._classifier = BaggingClassifier(ExtraTreesClassifier())

    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
