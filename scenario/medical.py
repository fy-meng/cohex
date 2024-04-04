import numpy as np
from sklearn.tree import DecisionTreeClassifier


class MedicalScenario:
    def __init__(self):
        rng = np.random.default_rng(0)

        n_samples = 200
        age_prob = [0.14, 0.15, 0.14, 0.15, 0.15, 0.11, 0.07, 0.06, 0.02, 0.01]
        age = rng.choice(10 * np.arange(10), size=n_samples, p=age_prob) + rng.uniform(0, 10, n_samples)
        age = age / 100
        history = rng.uniform(0, 1, size=n_samples)

        self.X = np.array([age, history]).T

        threshold = (4 * (age / 100) ** 2 + (0.75 * history) ** 2) > 0.4
        self.y = np.zeros(self.X.shape[0])
        self.y[threshold] = rng.uniform(0, 1, np.count_nonzero(threshold)) < 0.8
        self.y[np.logical_not(threshold)] = rng.uniform(0, 1, np.count_nonzero(1 - threshold)) < 0.2

        self.model = DecisionTreeClassifier(max_depth=2)
        self.model.fit(self.X, self.y)
        self.model.tree_.threshold[0] = 0.18807212
        self.model.tree_.threshold[1] = 0.80290332
        self.model.tree_.threshold[4] = 0.28563996
