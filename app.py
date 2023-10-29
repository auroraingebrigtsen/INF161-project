from flask import Flask, render_template, request
from waitress import serve
import pickle
import numpy as np
import pandas as pd
import holidays

app = Flask(__name__)

model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('./index.html')

@app.route('/predict', methods=['POST'])
def predict():
    ''' 
    '''
    features = dict(request.form)

    numerical_inputs = ["solskinstid", "lufttemperatur", "vindstyrke", "lufttrykk", "vindkast", "globalstraling", "vindretning"]

    def to_numeric(key, value, numeric_inputs = numerical_inputs):
        if key not in numeric_inputs:
            return value
        try:
            return float(value)
        except:
            return np.nan
    
    features = {key: to_numeric(key, value) for key, value in features.items()}

    def red_day(date) -> bool:
        norwegian_reddays = holidays.Norway(years=range(2010, 2024))
        return date in norwegian_reddays
    
    # definere features ut fra dato og klokkeslett
    def datetime_features(feature_dict:dict) -> dict:
        dato = pd.to_datetime(feature_dict["dato"])
        feature_dict["Maaned"] = dato.month
        #feature_dict["Aarstall"] = dato.year
        feature_dict["Ukedag"] = dato.weekday()
        feature_dict["Rod_dag"] = red_day(dato)
        del feature_dict["dato"]
        return feature_dict

    features = datetime_features(features)



    # prepare for prediction
    features_df = pd.DataFrame(features, index=[0])
    print(features_df)


    # sjekk input
    #if features_df.loc[0, 'LotArea'] <= 0:
    #    return render_template('./index.html',
    #                           prediction_text='LotArea must be positive')

    # predict
    prediction = model.predict(features_df)
    prediction = np.round(prediction[0])
    prediction = np.clip(prediction, 0, np.inf)

    # prepare output
    return render_template('./index.html',
                           prediction=prediction)



if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=8080)