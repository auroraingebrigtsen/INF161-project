import numpy as np
import pandas as pd

# Read the CSV file with specified data types
df = pd.read_csv('trafikkdata.csv', sep=';', dtype={'Navn': str, 'Vegreferanse': str, 'Trafikkmengde': str})
print(df.head(10))