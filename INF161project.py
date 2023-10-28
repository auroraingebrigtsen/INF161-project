import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from selection import ModelSelector, FeatureSelector
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from preprocessing import merge_dfs
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from visualizations import barplot, correlations
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle

rfr_param_grid = {
    'model__random_state': [42],
    'model__n_estimators': [100, 200, 300],
    'model__max_depth': [10, 20, 30]
}

svr_param_grid = {
    'model__C': [1, 10, 100],
    'model__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
}

mlp_param_grid = {
    'model__hidden_layer_sizes': [(50, 50), (100, 50, 25), (50,)],  # Sizes of hidden layers
    'model__activation': ['relu', 'tanh', 'logistic'],  # Activation function
    'model__learning_rate': ['constant', 'adaptive'],  # Learning rate schedule
    'model__random_state': [42],  # Random seed for reproducibility
}




def main():
    df = merge_dfs()

    # Lagre 2023 data til senere
    data_2023 = df[df['Aarstall'] == 2023].drop(columns=['Trafikkmengde'])

    # Droppe kolonner der trafikkmengde er nan
    data = df[df['Aarstall'] != 2023]
    data = data.dropna(subset=['Trafikkmengde'])

    # Dele i features og target, og splitte datasettet i trenings-og testdata
    X = data.drop(columns=['Trafikkmengde'])
    y = data['Trafikkmengde']

    # Splitter dataen
    X_train, X_test, y_train, y_test = train_test_split(X, y ,shuffle=False, test_size=0.3)

    # Korrelasjonsmatrise
    train_df = pd.concat([X_train, y_train], axis=1)
    correlations(train_df)

    # Visualisere treningsdata


    # Sjekke om dataen er poisson-fordelt
    print(f'\nMean: {np.mean(y_train)}\nVariance: {np.var(y_train)}\n')

    # Lage en baseline modell
    baseline = DummyRegressor()
    baseline.fit(X_train, y_train)
    base_pred = baseline.predict(X_test)
    base_rmse = np.sqrt(mean_squared_error(y_test, base_pred))
    print(f'Baseline model got RMSE: {base_rmse}')


    # Model selection
    imputers = [SimpleImputer(strategy='mean'), KNNImputer()]
    scalers = [MinMaxScaler(feature_range=(0,1)), StandardScaler()]
    model_selector = ModelSelector(X_train, y_train, imputers=imputers, scalers=scalers)
    model_selector.add_model(RandomForestRegressor(), rfr_param_grid)
    model_selector.add_model(SVR(), svr_param_grid)
    model_selector.add_model(MLPRegressor(), mlp_param_grid)
    best_model, best_score, best_imputer, best_scaler = model_selector.get_best()
    print(f'Best model is {best_model} with score: {best_score}')

    # Feature selection på den beste modellen
    features_to_check = ['Solskinstid', 'Lufttemperatur', 'Vindstyrke', 'Lufttrykk', 'Vindkast',
                         'Globalstraling', 'Vindretning']
    feature_selector = FeatureSelector(best_model, X_train, y_train, 3, features=features_to_check, imputer=best_imputer, scaler=best_scaler)
    feature_selector.fit()
    dropped_cols, reduced_X_train, best_model, best_imputer, best_scaler = feature_selector.get_best()
    print(f'Feature selector found that by dropping {dropped_cols} RMSE changes by {feature_selector.get_difference()}')
    print(f'Initial RMSE: {feature_selector.initial_score} \nNew RMSE: {feature_selector.best_score}')

    # Evaluering av den beste modellen
    reduced_X_test = X_test.drop(columns=dropped_cols)
    reduced_X_test = best_scaler.transform(reduced_X_test)
    X_test = pd.DataFrame(best_imputer.transform(reduced_X_test), columns=reduced_X_train.columns)
    best_pred = best_model.predict(X_test)
    best_rmse = np.sqrt(mean_squared_error(y_test, best_pred))
    print(f'Best model got RMSE: {best_rmse} on unseen data')

    #  2023 predictions
    dato_2023 = pd.Series(data_2023.index.date)
    tid_2023 = pd.Series(data_2023.index.time)
    data_2023 = pd.DataFrame(best_imputer.transform(data_2023.drop(columns=dropped_cols)), columns=reduced_X_train.columns)
    pred_2023 = pd.Series(best_model.predict(data_2023)).astype(int)
    result_2023 = pd.DataFrame({'Dato': dato_2023, 'Tid': tid_2023, 'Prediksjon': pred_2023})
    result_2023.to_csv('Predictions.csv', index=False)

    #  Lagre modellen og imputeren
    pickle.dump(best_model, open('model.pkl', 'wb'))
    pickle.dump(best_imputer, open('imputer.pkl', 'wb'))
    pickle.dump(best_scaler, open('scaler.pkl', 'wb'))

if __name__ == '__main__':
    main()