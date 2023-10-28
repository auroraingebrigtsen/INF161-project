import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from itertools import combinations

class ModelSelector:
    def __init__(self, X_train:pd.DataFrame, y_train:pd.Series, imputers:list, scalers:list, scoring='neg_mean_squared_error', splits=5):
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.splits = splits
        self.imputers = imputers
        self.scalers = scalers
        self.best_estimator = None
        self.best_score = 100 # dete kan være baseline score
        self.best_params = None
        self.best_imputer = None
    
    def tune(self, model, params:dict):
        """TODO"""
        for imputer in self.imputers:
            for scaler in self.scalers:
                pipeline = Pipeline([
                    ('imputer', imputer),
                    ('scaler', scaler),
                    ('model', model)
                ])
                cv = TimeSeriesSplit(n_splits=self.splits)
                grid_search = GridSearchCV(pipeline, params, cv=cv, scoring=self.scoring)
                cv_results = grid_search.fit(self.X_train, self.y_train)
                score = np.sqrt(-cv_results.best_score_) if self.scoring == 'neg_mean_squared_error' else cv_results.best_score_
                if score < self.best_score:
                    self.best_estimator = cv_results.best_estimator_.named_steps['model']
                    self.best_score = score
                    self.best_params = cv_results.best_params_
                    self.best_imputer = imputer
                    self.best_scaler = scaler


    def add_model(self, model, param_grid:dict) -> None:
        self.tune(model, param_grid)

    def get_best(self):
        """returns the best model"""
        imputer = self.best_imputer.fit(self.X_train)
        scaler = self.best_scaler.fit(self.X_train)
        self.X_train = self.best_imputer.transform(self.X_train)
        best_model = self.best_estimator.fit(self.X_train, self.y_train) # Fit on the whole training data
        return best_model, self.best_score, imputer, scaler
    

class FeatureSelector():
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame, max_combos:int, features:list, imputer, scaler, splits:int=5) -> None:
        self.model = model
        self.X = X_train
        self.y = y_train
        self.splits = splits
        self.imputer = imputer
        self.scaler = scaler
        self.max_combos = max_combos
        self.features = features # list containing names of the features you want to select from
        self.features_to_drop = None # name of the optimal combination of features
        self.best_score = 10000 # Her kan jeg ha initial score ? TODO
        self.initial_score = 10000

    def test_performance(self, X_variant):
        """Performs K-fold to test performance of a dataframe"""
        cv = TimeSeriesSplit(n_splits=self.splits)
        scores = cross_val_score(self.model, X_variant, self.y, cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-np.mean(scores))


    def find_combos(self):
        """

        """
        combo_list = []
        for i in range(self.max_combos):
            combinations_list = list(combinations(self.features, i+1))
            for combination in combinations_list:
                combo_list.append(list(combination))
        return combo_list


    def fit(self) -> None:
        """

        """
        features = self.X.columns
        self.X = self.imputer.transform(self.X)
        self.X = self.scaler.transform(self.X)
        self.X = pd.DataFrame(self.X, columns=features)
        self.initial_score = self.test_performance(self.X)
        combo_list = self.find_combos()
        for combination in combo_list:
            X_copy = self.X.copy()
            X_copy.drop(columns=combination, inplace=True)
            score = self.test_performance(X_copy)
            if score < self.best_score:
                self.best_score = score
                self.features_to_drop = combination
    
    def get_best(self):
        self.X = self.X.drop(columns=self.features_to_drop)
        self.imputer = self.imputer.fit_transform(self.X)
        self.scaler = self.scaler.fit_transform(self.X)
        self.model.fit(self.X, self.y)
        return self.features_to_drop, self.X, self.model, self.imputer, self.scaler
    
    def get_difference(self) -> float:
        return self.best_score - self.initial_score
    
"""
def scaler_selector(scalers, models, X_train, y_train, base_rmse=10000, splits=5):
    best_rmse = base_rmse
    best_scaler = None
    
    for scaler in scalers:
        cv = TimeSeriesSplit(n_splits=splits)
        total_rmse = [np.sqrt(-np.mean(cross_val_score(model, scaler.fit_transform(X_train), y_train, cv=cv, scoring='neg_mean_squared_error'))) for model in models]
        mean_rmse = np.mean(total_rmse)
        print(f'Using {type(scaler).__name__}, Models got mean RMSE: {mean_rmse}')
        if mean_rmse < best_rmse:
            best_rmse = mean_rmse
            best_scaler = scaler

    print(f'The best scaler is {type(best_scaler).__name__} with RMSE: {best_rmse}')
    return best_scaler
"""