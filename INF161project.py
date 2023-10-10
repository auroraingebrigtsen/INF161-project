import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error

# Read the CSV file with specified data types
traffic_df = pd.read_csv('trafikkdata.csv', sep="[|;]", engine='python')

# Replace missing values with nan
traffic_df['Trafikkmengde'] = traffic_df['Trafikkmengde'].replace('-', np.nan)

# Drop unecessary columns
traffic_df = traffic_df[['Dato','Fra tidspunkt','Trafikkmengde']]

# Make a single DateTime column
traffic_df['Tidspunkt'] = pd.to_datetime(traffic_df['Dato'] + ' ' + traffic_df['Fra tidspunkt'])

traffic_df = traffic_df[['Trafikkmengde', 'Tidspunkt']]

traffic_df.set_index('Tidspunkt', inplace=True)

folder_path = "weather_data"
csv_files = [f for f in os.listdir('weather_data/')]
weather_df = pd.DataFrame()

# Read weather data
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    weather_df = pd.concat([weather_df, df], ignore_index=True)
weather_df.head(50)

# Make Datetime column
weather_df['Tidspunkt'] = pd.to_datetime(weather_df['Dato'] + ' ' + weather_df['Tid'])
weather_df = weather_df.drop(columns=['Dato', 'Tid'])

weather_df.set_index('Tidspunkt', inplace=True)

# Change 9999 vals to NaN
weather_df = weather_df.replace(9999.99, np.nan)

# Make datetime column hourly instead of each 10 min
resampled_df = weather_df.resample('H').agg(
    {'Solskinstid':'sum', 'Lufttemperatur': 'mean', 'Vindstyrke': 'mean', 'Lufttrykk': 'mean', 'Vindkast': 'mean', 'Globalstraling': 'mean', 'Vindretning': 'mean' })

merged_df = traffic_df.merge(resampled_df, left_index=True, right_index=True) 

# split data

#test different models

#har ikke kommet så langt enda..