import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV


class ModelSelector:
    def __init__(self, X_train:pd.DataFrame, y_train:pd.Series, scoring='neg_mean_squared_error', splits=5):
        self.models = []
        self.X_train = X_train
        self.X_test = y_train
        self.scoring = scoring
        self.splits = splits
        self.best_estimator = None
        self.best_score = -1
        self.best_params = None
    
    def tune(self, model, params:dict):
        cv = TimeSeriesSplit(n_splits=self.splits)
        grid_search = GridSearchCV(model, params, cv=cv, scoring=self.scoring)
        score = np.sqrt(-grid_search.best_score_) if self.scoring == 'neg_mean_squared_error' else grid_search.best_score_
        if score > self.best_score:
            self.best_estimator = grid_search.best_estimator_
            self.best_score = np.sqrt(-grid_search.best_score_)
            self.best_params = grid_search.best_params_

    def add_model(self, model, param_grid:dict) -> None:
        self.tune(model, param_grid)
        self.models.append(model)
        
    def get_best(self) -> None:
        """returns the best model"""
        return self.best_estimator, self.best_score, self.best_params

    