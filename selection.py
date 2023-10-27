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
    def __init__(self, X_train:pd.DataFrame, y_train:pd.Series, imputers:list, scoring='neg_mean_squared_error', splits=5):
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.splits = splits
        self.imputers = imputers
        self.best_estimator = None
        self.best_score = 100 # dete kan være baseline score
        self.best_params = None
        self.best_imputer = None
    
    def tune(self, model, params:dict):
        """TODO"""
        if len(self.imputers) == 0:
            # Handle missing values using a default imputer strategy
            self.imputers.append(SimpleImputer(strategy='mean'))
        for imputer in self.imputers:
            pipeline = Pipeline([
                ('imputer', imputer),
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


    def add_model(self, model, param_grid:dict) -> None:
        self.tune(model, param_grid)

    def get_best(self):
        """returns the best model"""
        imputer = self.best_imputer.fit(self.X_train)
        self.X_train = self.best_imputer.transform(self.X_train)
        best_model = self.best_estimator.fit(self.X_train, self.y_train) # Fit on the whole training data
        return best_model, self.best_score, imputer
    

class FeatureSelector():
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame, max_combos:int, features:list, splits:int=5) -> None:
        self.model = model
        self.X = X_train
        self.y = y_train
        self.splits = splits
        self.max_combos = max_combos
        self.features = features # list containing names of the features you want to select from
        self.features_to_drop = None # name of the optimal combination of features
        self.best_score = 10000

    def test_performance(self, X_variant):
        """Performs K-fold to test performance of a dataframe"""
        cv = TimeSeriesSplit(n_splits=self.splits)
        scores = cross_val_score(self.model, X_variant, self.y, cv=cv, scoring='neg_mean_squared_error')
        return np.sqrt(-np.mean(scores))


    def find_combos(self):
        """
        0. create a list 
        1. loops trough n_combos
        2. append to list all combos of size of loop+1 (0,1,2)
        3. return list
        """
        combo_list = []
        for i in range(self.max_combos):
            combinations_list = list(combinations(self.features, i))
            for combination in combinations_list:
                combo_list.append(list(combination))
        return combo_list


    def fit(self) -> None:
        """
        1. runds find combos
        2. loops trough all elements in find combos list 
        3. create a copy of X
        4. remove the columns from the copy
        5. test performance on the new df
        6. if lower than best_SCORE save
        """
        combo_list = self.find_combos()
        for combination in combo_list:
            X_copy = self.X.copy()
            X_copy.drop(columns=combination, inplace=True)
            score = self.test_performance(X_copy)
            if score < self.best_score:
                self.best_score = score
                self.features_to_drop = combination
    
    def get_best(self):
        new_X = self.X.drop(columns=self.features_to_drop)
        return self.features_to_drop, new_X