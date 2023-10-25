import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

class ModelSelector:
    def __init__(self, X_train:pd.DataFrame, y_train:pd.Series, imputers:list, scoring='neg_mean_squared_error', splits=5):
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.splits = splits
        self.imputers = imputers
        self.best_estimator = None
        self.best_score = 100
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