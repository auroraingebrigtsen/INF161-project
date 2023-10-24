import os
import numpy as np
import pandas as pd
import holidays
from sklearn.model_selection import TimeSeriesSplit


def prepare_trafficdata():
    traffic_df = pd.read_csv('trafikkdata.csv', sep="[|;]", engine='python')

    print(f'Unike verdier i "Felt": {traffic_df["Felt"].unique()}')

    # Beholde kun "Totalt"-rader, for å unngå duplikater
    traffic_df = traffic_df[traffic_df['Felt'] == 'Totalt']

    # Sette nan på manglende verdier
    traffic_df['Trafikkmengde'] = traffic_df['Trafikkmengde'].replace('-', np.nan)

    # Droppe unødvendige kolonner
    traffic_df = traffic_df[['Dato','Fra tidspunkt','Trafikkmengde']]

    # Lage en datetime index
    traffic_df['Tidspunkt'] = pd.to_datetime(traffic_df['Dato'] + ' ' + traffic_df['Fra tidspunkt'])
    traffic_df = traffic_df[['Trafikkmengde', 'Tidspunkt']]
    traffic_df.set_index('Tidspunkt', inplace=True)

    # Endre trafikkmengde til float dtype
    traffic_df = traffic_df.astype({'Trafikkmengde':'float'})

    print(traffic_df.describe())
    return traffic_df


def prepare_weatherdata():
    folder_path = "weather_data"
    csv_files = [f for f in os.listdir('weather_data/')]
    weather_df = pd.DataFrame()

    # Lese værdata
    for csv_file in csv_files:
        file_path = os.path.join(folder_path, csv_file)
        df = pd.read_csv(file_path)
        weather_df = pd.concat([weather_df, df], ignore_index=True)

    proportion = weather_df[weather_df["Dato"] >= "2022-01-01"].shape[0] / weather_df.shape[0]
    print(f'Andel av værdata som inneholder "Relativ Luftfuktighet: {proportion}')
    print(weather_df.describe())

    start_vindkast = weather_df[weather_df["Vindkast"].notna()]
    print(f'"Vindkast" ble først målt {start_vindkast["Dato"].iloc[0]}')

    weather_df = weather_df.drop(columns=['Relativ luftfuktighet'])

    # Lage Datetime column
    weather_df['Tidspunkt'] = pd.to_datetime(weather_df['Dato'] + ' ' + weather_df['Tid'])
    weather_df = weather_df.drop(columns=['Dato', 'Tid'])
    weather_df.set_index('Tidspunkt', inplace=True)

    # Endre 9999 vals til NaN
    weather_df = weather_df.replace(9999.99, np.nan)

    # Make datetime column hourly instead of each 10 min
    resampled_df = weather_df.resample('H').agg(
        {'Solskinstid':'sum', 'Lufttemperatur': 'mean', 'Vindstyrke': 'mean', 'Lufttrykk': 'mean', 'Vindkast': 'mean', 'Globalstraling': 'mean', 'Vindretning': 'mean' })
    
    # Sjekke at nye verdier gir mening, ex. at max solskinstid <= 60
    print(resampled_df.describe())
    return resampled_df

def merge_dfs():
    traffic_df = prepare_trafficdata()
    weather_df = prepare_weatherdata()
    merged_df = traffic_df.merge(weather_df, left_index=True, right_index=True)

    # Droppe duplikate tidspunkt
    groups = merged_df.groupby(level=merged_df.index.names)
    merged_df = groups.last()

    # Splitte datetime kolonnen
    merged_df['Ukedag'] = merged_df.index.weekday
    merged_df["Maaned"] = merged_df.index.month
    merged_df["Aarstall"] = merged_df.index.year
    merged_df["Klokkeslett"] = merged_df.index.hour

    norske_helligdager = holidays.Norway(years=range(2010, 2024))
    merged_df["Rod_dag"] = merged_df.index.map(lambda x: int(x in norske_helligdager))

    merged_df.reset_index(drop=True, inplace=True)

    merged_df = merged_df.dropna(subset=['Trafikkmengde'])
    return merged_df