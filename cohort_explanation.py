import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeRegressor

from explainer import Explainer
from supervised_clustering import SRIDHCR


def cohex(explainer: Explainer, dataset: np.ndarray, n_cohorts: int, n_iter: int, termination_count: int,
          verbose=False, return_penalty=False, return_centroids=False):
    """
    Running CohEx.
    :param explainer: An `Explainer` instance. Should implement the
    `explain(dataset)` function as defined in `explainer.py`.
    :param dataset: np.ndarray, shape (n, f), where `n` is the number of
    samples, and `f` is the number of features.
    :param n_cohorts: _Expected_ number of cohorts.
    :param n_iter: Number of initializations to run iterative cohort
    explanation algorithms.
    :param termination_count: In each iteration, if the number of iteration
    without improvement exceeds this number, then the iteration would terminate.
    :param verbose: Whether to print debug information.
    :param return_penalty: If True, then the supervised clustering penalty is
    also returned. Incompatible with `return_centroids`.
    :param return_centroids: If True, then the centroids are also returned.
    Incompatible with `return_penalty`.
    :return k: int, the number of clusters.
    :return labels: np.ndarray, shape (n,), where each entry is an integer
    in [0, k).
    :return cohort_importance: np.ndarray, shape (k, f), the average importance
    of each cohort.
    """
    if isinstance(dataset, pd.DataFrame):
        dataset = dataset.values

    n_instances = dataset.shape[0]
    importance = explainer.explain(dataset)

    clustering = SRIDHCR(n_clusters=n_cohorts)
    centroids = dataset[np.random.choice(n_instances, n_cohorts, replace=False)]
    labels = clustering.assignment(dataset, centroids)

    penalty_best = float('inf')
    centroids_best = None
    labels_best = None
    importance_best = None

    for t in range(n_iter):
        penalty_iter_best = float('inf')
        centroids_iter_best = None
        labels_iter_best = None
        importance_iter_best = None

        n_iter_no_improvement = 0
        i = 0
        while True:
            i += 1

            for j in range(centroids.shape[0]):
                indices = np.where(labels == j)[0]
                cohort = dataset[indices]
                cohort_values = explainer.explain(cohort)
                importance[indices] = cohort_values

            labels = clustering.fit_predict(dataset, importance)
            centroids = clustering.cluster_centers_

            penalty = clustering.penalty(dataset, importance, centroids)
            if verbose:
                print(f'iter {t}.{i}:\n\tpenalty={penalty:.6f}')

            if penalty < penalty_iter_best:
                centroids_iter_best = centroids
                labels_iter_best = labels
                penalty_iter_best = penalty
                importance_iter_best = importance
                n_iter_no_improvement = 0
            else:
                n_iter_no_improvement += 1
                if n_iter_no_improvement >= termination_count:
                    break

        if penalty_iter_best < penalty_best:
            centroids_best = centroids_iter_best
            labels_best = labels_iter_best
            penalty_best = penalty_iter_best
            importance_best = importance_iter_best

    # compute avg importance for each cohort
    k = centroids_best.shape[0]
    shape = list(importance.shape)
    shape[0] = k
    cohort_importance = np.zeros(shape)
    for j in range(k):
        cohort_importance[j] = np.mean(importance_best[labels_best == j], axis=0)

    if return_penalty:
        return k, labels_best, cohort_importance, penalty_best
    elif return_centroids:
        return k, labels_best, cohort_importance, centroids_best
    else:
        return k, labels_best, cohort_importance


def hierarchical_exp(explainer, dataset, n_cohorts, index=None, return_centroids=False):
    importance = explainer.explain(dataset)

    clustering = SRIDHCR(n_clusters=n_cohorts, beta=5.0)
    # only consider index in certain columns
    if index is not None:
        labels = clustering.fit_predict(dataset[:, index], importance)
    else:
        labels = clustering.fit_predict(dataset, importance)
    centroids = clustering.cluster_centers_

    # compute avg importance for each cohort
    k = centroids.shape[0]
    shape = list(importance.shape)
    shape[0] = k
    cohort_importance = np.zeros([k] + list(importance.shape[1:]))
    for j in range(k):
        cohort_importance[j] = np.mean(importance[labels == j], axis=0)

    if return_centroids:
        return k, labels, cohort_importance, centroids
    else:
        return k, labels, cohort_importance


def kmeans_by_feature(explainer, dataset, n_cohorts):
    clustering = KMeans(n_clusters=n_cohorts, n_init='auto')
    importance = explainer.explain(dataset)

    labels = clustering.fit_predict(dataset.reshape(dataset.shape[0], -1))
    shape = list(importance.shape)
    shape[0] = n_cohorts
    cohort_importance = np.zeros(shape)
    for j in range(n_cohorts):
        cohort_importance[j] = np.mean(importance[labels == j], axis=0)
    return n_cohorts, labels, cohort_importance


def vine(explainer, dataset, n_cohorts):
    clustering = KMeans(n_clusters=n_cohorts, n_init='auto')
    importance = explainer.explain(dataset)
    labels = clustering.fit_predict(importance.reshape(importance.shape[0], -1))
    shape = list(importance.shape)
    shape[0] = n_cohorts
    cohort_importance = np.zeros(shape)
    for j in range(n_cohorts):
        cohort_importance[j] = np.mean(importance[labels == j], axis=0)
    return n_cohorts, labels, cohort_importance


def repid(explainer, dataset, max_depth):
    clustering = DecisionTreeRegressor(max_depth=max_depth)
    importance = explainer.explain(dataset)
    x = dataset.reshape(dataset.shape[0], -1)
    y = importance.reshape(importance.shape[0], -1)
    clustering.fit(x, y)
    labels = clustering.apply(x)  # not necessarily between 0 and (n_cohorts - 1)
    # convert labels to be in range 0 and (n_cohorts - 1)
    label_set = np.unique(labels)
    labels = np.array([np.where(label_set == i)[0][0] for i in labels])

    n_cohorts = clustering.get_n_leaves()
    shape = list(importance.shape)
    shape[0] = n_cohorts
    cohort_importance = np.zeros(shape)
    for j in range(n_cohorts):
        cohort_importance[j] = np.mean(importance[labels == j], axis=0)
    return n_cohorts, labels, cohort_importance
