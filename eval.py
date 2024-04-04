import numpy as np
import matplotlib.pyplot as plt


def _in_arr(row: np.ndarray, arr: np.ndarray):
    return any((arr[:] == row).all(axis=1))


class AlternateExplainee:
    def __init__(self, model, cohort, classes, p=0.05):
        self.model = model
        self.cohort = cohort
        self.classes = classes
        self.p = p

    def predict(self, X):
        if len(X.shape) == 1:
            if _in_arr(X, self.cohort):
                return self.model.predict(X)
            else:
                if np.random.rand() > self.p:
                    return self.model.predict(X)
                else:
                    return np.random.choice(self.classes)
        else:
            pred_actual = self.model.predict(X)
            pred_rand = np.random.choice(self.classes, X.shape[0])
            idx = np.logical_and(
                [not _in_arr(x, self.cohort) for x in X],
                np.random.rand(X.shape[0]) < self.p
            )
            return np.choose(idx, [pred_actual, pred_rand])


def locality(explainer_class, model, dataset, labels, importance, classes, n_iter=10, **explainer_kwargs):
    k = len(np.unique(labels))

    loss = 0.0
    for j in range(k):
        cohort = dataset[labels == j]
        for t in range(n_iter):
            model_alt = AlternateExplainee(model, cohort, classes)
            explainer = explainer_class(model_alt, **explainer_kwargs)
            importance_alt = np.mean(explainer.explain(cohort), axis=0)
            loss += np.sum((importance_alt - importance[j]) ** 2) / k / n_iter

    return loss


def stability_importance(explainer, dataset, labels, importance, n_iter=10, recompute=False):
    k = len(np.unique(labels))

    importance_all = explainer.explain(dataset)

    loss = 0.0
    for j in range(k):
        cohort = dataset[labels == j]
        non_cohort = dataset[labels != j]
        for t in range(n_iter):
            sample_idx = np.random.choice(non_cohort.shape[0])
            if recompute:
                importance_alt = np.mean(importance_all[list(labels == j) + [sample_idx]])
            else:
                sample = non_cohort[sample_idx]
                cohort_alt = np.vstack((cohort, sample[np.newaxis]))
                importance_alt = np.mean(explainer.explain(cohort_alt), axis=0)
            loss += np.sum((importance_alt - importance[j]) ** 2) / k / n_iter

    return loss
