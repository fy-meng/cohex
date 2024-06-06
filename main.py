import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from tqdm import trange

from cohort_explanation import cohex, hierarchical_exp, vine, kmeans_by_feature, repid
from scenario.medical import MedicalScenario
from scenario.MNIST import MNISTScenario
from scenario.bike_sharing import BikeScenario
from scenario.compas import COMPAS
from explainer import SHAPExplainer, LIMEExplainer
from eval import locality, stability_importance
from supervised_clustering import SRIDHCR


# ======== Medical motivational scenario ========
def medical_visualize(method, save_dir='output/medical_lime', explainer_type='lime', plot_distribution=False):
    scenario = MedicalScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular', kernel_width=3)
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    # plot distribution of importance
    if plot_distribution:
        importance = explainer.explain(scenario.X)
        plt.figure(dpi=300)
        plt.scatter(importance[:, 0], importance[:, 1], s=3)
        plt.xlabel('importance of age')
        plt.ylabel('importance of family history')
        plt.axis('equal')
        plt.savefig(os.path.join(save_dir, 'importance_distribution.png'))
        plt.clf()

    if method == 'kmeans_by_feature':
        k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
    elif method == 'cohex':
        k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4,
                                      n_iter=5, termination_count=5, verbose=False)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    if len(importance.shape) > 2:
        importance = importance[:, :, 0]

    fig, ax = plt.subplots(dpi=300)
    for i in range(k):
        plt.scatter(scenario.X[labels == i, 0], scenario.X[labels == i, 1], marker='.')
        plt.text(np.mean(scenario.X[labels == i, 0]), np.mean(scenario.X[labels == i, 1]),
                 f'({importance[i, 0]:.2f}, {importance[i, 1]:.2f})',
                 color=plt.get_cmap('tab10')(i), bbox=dict(facecolor='white', alpha=0.75))
    plt.axvline(scenario.model.tree_.threshold[0], c=plt.get_cmap('tab10')(3), label='1st split')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1], ['0', '20', '40', '60', '80', '100'])
    xmin, xmax = ax.get_xlim()
    xloc = (scenario.model.tree_.threshold[0] - xmin) / (xmax - xmin)
    plt.axhline(scenario.model.tree_.threshold[1], xmax=xloc, color=plt.get_cmap('tab10')(4), ls='--',
                label='2nd splits')
    plt.axvline(scenario.model.tree_.threshold[4], color=plt.get_cmap('tab10')(4), ls='--')
    plt.xlabel('age')
    plt.ylabel('family history')
    plt.savefig(os.path.join(save_dir, f'{method}.png'))

    np.save(os.path.join(save_dir, f'{method}_labels.npy'), labels)
    np.save(os.path.join(save_dir, f'{method}_importance.npy'), importance)


def medical_locality(method, n_iter=10, explainer_type='lime'):
    scenario = MedicalScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = locality(LIMEExplainer, scenario.model, scenario.X, labels, importance, classes=[0, 1],
                             mode='tabular', num_classes=2)

    print('locality')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def medical_stability_importance(method, n_iter=10, explainer_type='lime'):
    scenario = MedicalScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = stability_importance(explainer, scenario.X, labels, importance)

    print('importance stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def medical_stability_cohort(method, n_iter=10, explainer_type='lime'):
    scenario = MedicalScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    if method == 'kmeans_by_feature':
        _, labels, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        _, labels, _ = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        _, labels, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
    elif method == 'cohex':
        _, labels, _ = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                             verbose=False)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if explainer_type == 'lime':
            explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular', kernel_width=0.4)
        elif explainer_type == 'shap':
            explainer = SHAPExplainer(scenario.model, mode='default')
        elif explainer_type == 'kernel_shap':
            explainer = SHAPExplainer(scenario.model, mode='kernel')
        else:
            raise ValueError

        if method == 'kmeans_by_feature':
            _, labels_alt, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            _, labels_alt, _ = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            _, labels_alt, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            _, labels_alt, _ = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                     verbose=False)
        elif method == 'repid':
            _, labels_alt, _ = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError

        losses[i] = adjusted_rand_score(labels, labels_alt)

    print('cohort stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


# also outputs penalty evaluation
def medical_num_cohorts(method, explainer_type='lime'):
    scenario = MedicalScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    base_importance = explainer.explain(scenario.X)

    if method == 'repid':
        # special handling repid
        # -1 represent N/A
        penalties = [[] for _ in range(17)]
        for max_depth in trange(2, 5):
            for t in range(10):
                k, labels, _ = repid(explainer, scenario.X, max_depth=max_depth)
                clustering = SRIDHCR(n_clusters=k)
                penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                penalties[k - 2].append(penalty)
        df = pd.DataFrame({i: pd.Series(value) for i, value in enumerate(penalties)})
        df.to_csv(f'output/medical_{explainer_type}/{method}_num_cohorts.csv')
        # showing mean and std at depth=2
        print('penalty:')
        print(f'    {np.mean(penalties[2])} +- {np.std(penalties[2])}')
    else:
        penalties = np.zeros((15, 10))
        for k in trange(2, 17):
            clustering = SRIDHCR(n_clusters=k)
            for t in range(10):
                if method == 'kmeans_by_feature':
                    _, labels, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'vine':
                    _, labels, _ = vine(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'hierarchical':
                    _, labels, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'cohex':
                    _, _, _, penalty = cohex(explainer, scenario.X, k, 5, 7,
                                             verbose=False, return_penalty=True)
                else:
                    raise ValueError
                penalties[k - 2, t] = penalty
        print('penalty:')
        print(f'    {np.mean(penalties[2])} +- {np.std(penalties[2])}')
        np.save(f'output/medical_{explainer_type}/{method}_num_cohorts.npy', penalties)


# ======== MNIST ========
def mnist_visualize(method, digits=(7, 9), n_samples=200, n_cohorts=4):
    dirname = os.path.join('output', f'mnist_{method}')
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    scenario = MNISTScenario(digits, n_samples)
    explainer = SHAPExplainer(scenario.model, mode='deep')
    if method == 'vine':
        k, labels, importance = vine(explainer, scenario.X, n_cohorts=n_cohorts)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=int(np.log2(n_cohorts)))
    elif method == 'hierarchical':
        k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=n_cohorts)
    elif method == 'cohex':
        k, labels, importance = cohex(explainer, scenario.X,
                                      n_cohorts=n_cohorts, n_iter=5, termination_count=5)
    np.save(os.path.join(dirname, 'labels.npy'), labels)
    np.save(os.path.join(dirname, 'importance.npy'), importance)

    # plot cohort definitions and samples
    fig, ax = plt.subplots(k, 10, dpi=300, figsize=(8, 4))
    for j in range(k):
        for l in range(10):
            ax[j, l].axis('off')
    for j in range(k):
        indices = np.where(labels == j)[0]
        if len(indices) > 0:
            for l in range(min(10, len(indices))):
                ax[j, l].imshow(scenario.X[indices[l]].reshape((28, 28)), cmap='gray')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.01)
    plt.savefig(os.path.join(dirname, 'cohorts.png'))

    # plot explanation for each cohort
    max_val = max(
        abs(np.mean(importance) + 2 * np.std(importance)),
        abs(np.mean(importance) - 2 * np.std(importance))
    )
    for j in range(k):
        fig, ax = plt.subplots(2, 5, dpi=300, figsize=(5, 2))
        for l in range(5):
            ax[0, l].axis('off')
            ax[1, l].axis('off')
            ax[0, l].imshow(importance[j, l], cmap='gray', vmin=-max_val, vmax=max_val)
            ax[1, l].imshow(importance[j, l + 5], cmap='gray', vmin=-max_val, vmax=max_val)
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1, wspace=0.01, hspace=0.004)
        plt.savefig(os.path.join(dirname, f'c{j}.png'))
        plt.clf()
        plt.close(fig)


def mnist_locality(method, n_iter=10):
    scenario = MNISTScenario()

    explainer = SHAPExplainer(scenario.model, mode='deep')

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = locality(SHAPExplainer, scenario.model, scenario.X, labels, importance,
                             classes=[0, 1], mode='deep')

    print('locality')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def mnist_penalty(method, n_iter=10):
    scenario = MNISTScenario()

    explainer = SHAPExplainer(scenario.model, mode='deep')
    base_importance = explainer.explain(scenario.X)

    penalties = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError

        clustering = SRIDHCR(n_clusters=k)
        penalties[i] = clustering.penalty_labels(scenario.X, base_importance, labels)

    print('penalty')
    print(f'    {np.mean(penalties)} +- {np.std(penalties)}')


def mnist_stability_importance(method, n_iter=10):
    scenario = MNISTScenario(test_size=200)
    explainer = SHAPExplainer(scenario.model, mode='deep')

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = stability_importance(explainer, scenario.X, labels, importance)

    print('importance stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def mnist_stability_cohort(method, n_iter=10):
    scenario = MNISTScenario()

    explainer = SHAPExplainer(scenario.model, mode='deep')
    if method == 'kmeans_by_feature':
        k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
    elif method == 'cohex':
        k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                      verbose=False)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        explainer = SHAPExplainer(scenario.model, mode='deep')
        if method == 'kmeans_by_feature':
            k, labels_alt, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels_alt, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels_alt, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels_alt, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                              verbose=False)
        elif method == 'repid':
            k, labels_alt, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = adjusted_rand_score(labels, labels_alt)

    print('cohort stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


# ======== bike sharing ========
def bike_visualize(method, save_dir='output/bike_shap', explainer_type='shap'):
    scenario = BikeScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular', kernel_width=3)
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    if method == 'kmeans_by_feature':
        k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        k, labels, importance, centroids = hierarchical_exp(explainer, scenario.X, n_cohorts=4, return_centroids=True)
        np.save(os.path.join(save_dir, f'{method}_centroids.npy'), centroids)
    elif method == 'cohex':
        k, labels, importance, centroids = cohex(explainer, scenario.X, n_cohorts=4,
                                                 n_iter=5, termination_count=5, verbose=False, return_centroids=True)
        np.save(os.path.join(save_dir, f'{method}_centroids.npy'), centroids)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    np.save(os.path.join(save_dir, f'{method}_labels.npy'), labels)
    np.save(os.path.join(save_dir, f'{method}_importance.npy'), importance)

    for i in range(k):
        print(f'cohort {i}')
        print(f'importance:')
        print(importance[i])


def bike_locality(method, n_iter=10, explainer_type='shap'):
    scenario = BikeScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = locality(LIMEExplainer, scenario.model, scenario.X, labels, importance, classes=[0, 1],
                             mode='tabular', num_classes=2)

    print('locality')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def bike_stability_importance(method, n_iter=10, explainer_type='shap'):
    scenario = BikeScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = stability_importance(explainer, scenario.X, labels, importance)

    print('importance stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def bike_stability_cohort(method, n_iter=10, explainer_type='shap'):
    scenario = BikeScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    if method == 'kmeans_by_feature':
        _, labels, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        _, labels, _ = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        _, labels, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
    elif method == 'cohex':
        _, labels, _ = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                             verbose=False)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if explainer_type == 'lime':
            explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular', kernel_width=0.4)
        elif explainer_type == 'shap':
            explainer = SHAPExplainer(scenario.model.predict, mode='default')
        elif explainer_type == 'kernel_shap':
            explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
        else:
            raise ValueError

        if method == 'kmeans_by_feature':
            _, labels_alt, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            _, labels_alt, _ = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            _, labels_alt, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            _, labels_alt, _ = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                     verbose=False)
        elif method == 'repid':
            _, labels_alt, _ = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError

        losses[i] = adjusted_rand_score(labels, labels_alt)

    print('cohort stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def bike_num_cohorts(method, explainer_type='shap'):
    scenario = BikeScenario()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    base_importance = explainer.explain(scenario.X)

    if method == 'repid':
        # special handling repid
        # -1 represent N/A
        penalties = [[] for _ in range(17)]
        for max_depth in trange(2, 5):
            for t in range(10):
                k, labels, _ = repid(explainer, scenario.X, max_depth=max_depth)
                clustering = SRIDHCR(n_clusters=k)
                penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                penalties[k - 2].append(penalty)
        df = pd.DataFrame({i: pd.Series(value) for i, value in enumerate(penalties)})
        df.to_csv(f'output/bike_shap/{method}_num_cohorts.csv')
        # showing mean and std at depth=2
        print('penalty:')
        print(f'    {np.mean(penalties[2])} +- {np.std(penalties[2])}')
    else:
        penalties = np.zeros((15, 10))
        for k in trange(2, 17):
            clustering = SRIDHCR(n_clusters=k)
            for t in range(10):
                if method == 'kmeans_by_feature':
                    _, labels, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'vine':
                    _, labels, _ = vine(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'hierarchical':
                    _, labels, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'cohex':
                    _, _, _, penalty = cohex(explainer, scenario.X, k, 5, 7,
                                             verbose=False, return_penalty=True)
                else:
                    raise ValueError
                penalties[k - 2, t] = penalty
        print('penalty:')
        print(f'    {np.mean(penalties[2])} +- {np.std(penalties[2])}')
        np.save(f'output/bike_shap/{method}_num_cohorts.npy', penalties)


# ======== COMPAS ========
def compas_visualize(method, save_dir='output/compas_shap', explainer_type='shap'):
    scenario = COMPAS()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular', kernel_width=3)
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model, mode='kernel')
    else:
        raise ValueError

    if method == 'kmeans_by_feature':
        k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        k, labels, importance, centroids = hierarchical_exp(explainer, scenario.X, n_cohorts=4, return_centroids=True)
        np.save(os.path.join(save_dir, f'{method}_centroids.npy'), centroids)
    elif method == 'cohex':
        k, labels, importance, centroids = cohex(explainer, scenario.X, n_cohorts=4,
                                                 n_iter=5, termination_count=5, verbose=False, return_centroids=True)
        np.save(os.path.join(save_dir, f'{method}_centroids.npy'), centroids)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    np.save(os.path.join(save_dir, f'{method}_labels.npy'), labels)
    np.save(os.path.join(save_dir, f'{method}_importance.npy'), importance)

    for i in range(k):
        print(f'cohort {i}')
        print(f'importance:')
        print(importance[i])


def compas_locality(method, n_iter=10, explainer_type='shap'):
    scenario = COMPAS()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = locality(LIMEExplainer, scenario.model, scenario.X, labels, importance, classes=[0, 1],
                             mode='tabular', num_classes=2)

    print('locality')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def compas_stability_importance(method, n_iter=10, explainer_type='shap'):
    scenario = COMPAS()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if method == 'kmeans_by_feature':
            k, labels, importance = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            k, labels, importance = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            k, labels, importance = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            k, labels, importance = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                          verbose=False)
        elif method == 'repid':
            k, labels, importance = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError
        losses[i] = stability_importance(explainer, scenario.X, labels, importance)

    print('importance stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def compas_stability_cohort(method, n_iter=10, explainer_type='shap'):
    scenario = COMPAS()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    if method == 'kmeans_by_feature':
        _, labels, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
    elif method == 'vine':
        _, labels, _ = vine(explainer, scenario.X, n_cohorts=4)
    elif method == 'hierarchical':
        _, labels, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
    elif method == 'cohex':
        _, labels, _ = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                             verbose=False)
    elif method == 'repid':
        k, labels, importance = repid(explainer, scenario.X, max_depth=2)
    else:
        raise ValueError

    losses = np.zeros(n_iter)
    for i in trange(n_iter):
        if explainer_type == 'lime':
            explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular', kernel_width=0.4)
        elif explainer_type == 'shap':
            explainer = SHAPExplainer(scenario.model.predict, mode='default')
        elif explainer_type == 'kernel_shap':
            explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
        else:
            raise ValueError

        if method == 'kmeans_by_feature':
            _, labels_alt, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=4)
        elif method == 'vine':
            _, labels_alt, _ = vine(explainer, scenario.X, n_cohorts=4)
        elif method == 'hierarchical':
            _, labels_alt, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=4)
        elif method == 'cohex':
            _, labels_alt, _ = cohex(explainer, scenario.X, n_cohorts=4, n_iter=5, termination_count=5,
                                     verbose=False)
        elif method == 'repid':
            _, labels_alt, _ = repid(explainer, scenario.X, max_depth=2)
        else:
            raise ValueError

        losses[i] = adjusted_rand_score(labels, labels_alt)

    print('cohort stability')
    print(f'    {np.mean(losses)} +- {np.std(losses)}')


def compas_num_cohorts(method, explainer_type='shap'):
    scenario = COMPAS()

    if explainer_type == 'lime':
        explainer = LIMEExplainer(scenario.model, num_classes=2, mode='tabular')
    elif explainer_type == 'shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='default')
    elif explainer_type == 'kernel_shap':
        explainer = SHAPExplainer(scenario.model.predict, mode='kernel')
    else:
        raise ValueError

    base_importance = explainer.explain(scenario.X)

    if method == 'repid':
        # special handling repid
        # -1 represent N/A
        penalties = [[] for _ in range(17)]
        for max_depth in trange(2, 5):
            for t in range(10):
                k, labels, _ = repid(explainer, scenario.X, max_depth=max_depth)
                clustering = SRIDHCR(n_clusters=k)
                penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                penalties[k - 2].append(penalty)
        df = pd.DataFrame({i: pd.Series(value) for i, value in enumerate(penalties)})
        df.to_csv(f'output/bike_shap/{method}_num_cohorts.csv')
        # showing mean and std at depth=2
        print('penalty:')
        print(f'    {np.mean(penalties[2])} +- {np.std(penalties[2])}')
    else:
        penalties = np.zeros((15, 10))
        for k in trange(2, 17):
            clustering = SRIDHCR(n_clusters=k)
            for t in range(10):
                if method == 'kmeans_by_feature':
                    _, labels, _ = kmeans_by_feature(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'vine':
                    _, labels, _ = vine(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'hierarchical':
                    _, labels, _ = hierarchical_exp(explainer, scenario.X, n_cohorts=k)
                    penalty = clustering.penalty_labels(scenario.X, base_importance, labels)
                elif method == 'cohex':
                    _, _, _, penalty = cohex(explainer, scenario.X, k, 5, 7,
                                             verbose=False, return_penalty=True)
                else:
                    raise ValueError
                penalties[k - 2, t] = penalty
        print('penalty:')
        print(f'    {np.mean(penalties[2])} +- {np.std(penalties[2])}')
        np.save(f'output/bike_shap/{method}_num_cohorts.npy', penalties)


def main():
    # evaluating bike
    for method in ('vine', 'hierarchical', 'repid', 'cohex'):
        print(f'======== {method} ========')
        compas_locality(method, explainer_type='lime')
        compas_stability_cohort(method, explainer_type='lime')
        compas_stability_importance(method, explainer_type='lime')
        compas_num_cohorts(method, explainer_type='lime')


if __name__ == '__main__':
    main()
