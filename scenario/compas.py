import os

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from tempeh.datasets import CompasPerformanceDatasetWrapper
import xgboost as xgb


class COMPAS:
    def __init__(self):
        if not os.path.exists('./scenario/compas_data'):
            compas_dataset = CompasPerformanceDatasetWrapper.generate_dataset_class('compas')()
            self.X_train, self.X_test = compas_dataset.get_X(format=pd.DataFrame)
            self.y_train, self.y_test = compas_dataset.get_y(format=pd.Series)

            self.y_train = self.y_train.astype(int)
            self.y_test = self.y_test.astype(int)

            os.mkdir('./scenario/compas_data')
            self.X_train.to_csv('./scenario/compas_data/X_train.csv', index=False)
            self.X_test.to_csv('./scenario/compas_data/X_test.csv', index=False)
            self.y_train.to_csv('./scenario/compas_data/y_train.csv', index=False)
            self.y_test.to_csv('./scenario/compas_data/y_test.csv', index=False)
        else:
            self.X_train = pd.read_csv('./scenario/compas_data/X_train.csv')
            self.X_test = pd.read_csv('./scenario/compas_data/X_test.csv')
            self.y_train = pd.read_csv('./scenario/compas_data/y_train.csv')
            self.y_test = pd.read_csv('./scenario/compas_data/y_test.csv')

        self.feature_names = self.X_train.columns.tolist()

        model_path = './scenario/compas.json'
        if os.path.exists(model_path):
            self.model = xgb.XGBRegressor(enable_categorical=True, tree_method='hist')
            self.model.load_model(model_path)
        else:
            self.model = xgb.XGBClassifier(enable_categorical=True, tree_method='hist')
            self.train()
        self.model.save_model(model_path)

        self.X = self.X_test.values
        self.y = self.y_test.values

    def train(self):
        # Define the hyperparameters for tuning
        params = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 4, 5],
            'learning_rate': [0.1, 0.01, 0.001]
        }

        # perform hyperparameter tuning using gridsearch
        grid_search = GridSearchCV(self.model, params, cv=5, scoring='accuracy')
        grid_search.fit(self.X_train, self.y_train)

        print('Best hyperparameters:', grid_search.best_params_)

        self.model = grid_search.best_estimator_
        self.model.fit(self.X_train, self.y_train)

        y_pred = self.model.predict(self.X_test)

        acc = np.mean(y_pred == self.y_test)
        print(f'test acc = {acc:.2f}')


COMPAS()
