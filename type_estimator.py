from __future__ import annotations

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, \
    BaggingClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


class TypeEstimator:

    def __init__(self, est):
        # # Baseline
        # self._classifier = DecisionTreeClassifier()

        # f1 on train - 1, f1 on evaluation - 0.226
        # max type evaluation f1score: 0.283, with max_depth=11
        # self._classifier = RandomForestClassifier(max_depth=estimators)
        # f1 on evaluation - 0.278
        # self._classifier = RandomForestClassifier()

        # f1 on evaluation - 0.235
        # self._classifier = ExtraTreesClassifier(n_estimators=est)

        # f1 on train - 1, f1 on evaluation - 0.303 (1 neighbor)
        # self._classifier = KNeighborsClassifier(n_neighbors=estimators)

        # f1 on evaluation - 0.286
        # 180 est - 0.307
        self._classifier = GradientBoostingClassifier(n_estimators=est)


        # ff1 on evaluation - 0.246
        # self._classifier = BaggingClassifier(ExtraTreesClassifier())



    def fit_classifier(self, X, y):
        self._classifier.fit(X, y)

    def predict_classifier(self, X):
        return self._classifier.predict(X).reshape(-1, 1)

    def loss_classifier(self, X, y):
        return f1_score(y, self.predict_classifier(X), average='macro')
