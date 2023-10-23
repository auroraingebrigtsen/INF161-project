import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from model_selection import ModelSelector
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from Preprocessing import merge_dfs

rfr_param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

svr_param_grid = {
    'C': [1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

en_param_grid = {
    'alpha': [0.1, 0.5, 1.0],
    'l1_ratio': [0.1, 0.5, 0.9],
}



def main():
    df = merge_dfs()

    # Lagre 2023 data til senere
    data_2023 = df[df['Aarstall'] == 2023]

    data = df[df['Aarstall'] != 2023]
    X = data.drop(['Trafikkmengde'])
    y = data['Trafikkmengde']

    X_train, X_test, y_train, y_test = train_test_split(X, y ,shuffle=False, test_size=0.2)

    #  Create a baseline model
    baseline = DummyRegressor()
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    print(f'Baseline model got RMSE: {base_rmse}')

    model_selector = ModelSelector(X_train, y_train)
    model_selector.add_model(RandomForestRegressor(), rfr_param_grid)
    model_selector.add_model(SVR(), svr_param_grid)
    model_selector.add_model(ElasticNet(), en_param_grid)

    best_model, best_score, best_params = model_selector.best_estimator()
    best_pred = best_model.predict(X_test)
    best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    print(f'Best model is {best_model} with score: {best_rmse}\nModel has parameters: {best_params}')

if __name__ == '__main__':
    main()