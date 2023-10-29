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
    def __init__(self, X_train:pd.DataFrame, y_train:pd.Series, imputers:list, scalers:list, scoring='neg_mean_squared_error', splits:int=5) -> None:
        """
        Initialize a ModelSelector instance.

        Parameters:
        - X_train: pd.DataFrame, the feature data for training
        - y_train: pd.Series, the target data for training
        - imputers: list, a list of imputation techniques to be evaluated
        - scalers: list, a list of scaling methods to be evaluated
        - scoring: str, the scoring metric used for model evaluation (default is 'neg_mean_squared_error').
        - splits: int, the number of time series splits for cross-validation (default is 5).
        """
        self.X_train = X_train
        self.y_train = y_train
        self.scoring = scoring
        self.splits = splits
        self.imputers = imputers
        self.scalers = scalers
        self.best_estimator = None
        self.best_score = 10000
        self.best_params = None
        self.best_imputer = None
    
    def tune(self, model, params:dict):
        """
        Search for hyperparameters of a model using different imputers and scalers.
        Finds the optimal hyperparameters, imputation technique and scaler.

        Parameters:
        - model: the machine learning model to be tuned
        - params: dict, hyperparameters to include in search
        """
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
        """
        Add and tune a new machine learning model.

        Parameters:
        - model: the machine learning model to be added and tuned
        - param_grid: dict, hyperparameters to include in search
        """
        self.tune(model, param_grid)

    def get_best(self):
        """
        Get the results from the search.

        Returns:
        - best_model: the best-tuned machine learning model
        - best_score: the score of the best-tuned model
        - imputer: the best imputer used for preprocessing
        - scaler: the best feature scaler used for preprocessing
        """
        imputer = self.best_imputer.fit(self.X_train) # Fit imputer, scaler and model on whole training data
        scaler = self.best_scaler.fit(self.X_train)
        self.X_train = self.best_imputer.transform(self.X_train)
        self.X_train = self.best_scaler.transform(self.X_train)
        best_model = self.best_estimator.fit(self.X_train, self.y_train)
        return best_model, self.best_score, imputer, scaler
    

class FeatureSelector():
    def __init__(self, model, X_train: pd.DataFrame, y_train: pd.DataFrame, max_combos:int, features:list, imputer, scaler, splits:int=5) -> None:
        """
        Initialize a FeatureSelector instance.

        Parameters:
        - model: a machine learning model 
        - X_train: pd.DataFrame, the feature data for training
        - y_train: pd.Series, the target data for training
        - max_combos: int, the maximum number of feature combinations to explore.
        - features: list, feature names to be selected from
        - imputer: the imputer used for data preprocessing
        - scaler: the feature scaler used for data preprocessing
        - splits: int, the number of time series splits for cross-validation (default is 5)
        """
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
        """
        Performs K-fold cross-validation to test the performance of a feature set.

        Parameters:
        - X_variant: pd.DataFrame, the variant of input features to evaluate

        Returns:
        - The RMSE (Root Mean Squared Error) of the model's performance.
        """
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
        self.imputer = self.imputer.fit(self.X)
        self.scaler = self.scaler.fit(self.X)
        self.X = self.best_imputer.transform(self.X)
        self.X = self.best_scaler.transform(self.X)
        self.model.fit(self.X, self.y)
        return self.features_to_drop, self.X, self.model, self.imputer, self.scaler
    
    def get_difference(self) -> float:
        return self.best_score - self.initial_score
    
