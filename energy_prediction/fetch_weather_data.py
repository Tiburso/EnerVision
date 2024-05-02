# %%
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import requests
from glob import glob

from pvlib import pvsystem, modelchain, location, irradiance
from pvlib.solarposition import get_solarposition

# %%
def get_hourly_weather_data_for_pvlib(stations, start_date, end_date, timezone = 'UTC'):
    '''
    Function to get hourly weather variables T (temperature) and
    Q (global radiation) from KNMI 
    
    Args: 
        stations   (str): NMI-stations separated by ':' 
        start_date (str): start date, format yyyymmdd
        end_date   (str): end date (included), format yyyymmdd
        timezone   (str, optional): timezone

    Returns:
        df: DataFrame with DateTime-index, columns T (temp), Q (global radiation) 
    '''

    url = 'https://www.daggegevens.knmi.nl/klimatologie/uurgegevens'

    data = {
        'start': start_date,
        'end': end_date,
        'vars': 'Q:T:FH',
        'stns': stations,
        'fmt': 'json'
        }

    response = requests.post(url, data = data)    
    weather_df = pd.DataFrame(response.json())
    # correct units
    weather_df['T'] = weather_df['T'] / 10          # is in 0.1 degrees C, to degrees C
    weather_df['Q'] = weather_df['Q'] * (1 / 0.36)  # is in J/m2, to W / m2
    weather_df['FH'] = weather_df['FH']             
    
    # create date_time index, convert timezone
    weather_df['hour'] = weather_df['hour'] - 1     # is from 1-24, to 0-23
    weather_df['date_time'] = pd.to_datetime(weather_df['date']) + pd.to_timedelta(weather_df['hour'].astype(int), unit='h')
    weather_df.index = weather_df.date_time
    weather_df = weather_df.drop(['station_code', 'date', 'hour', 'date_time'], axis = 1)
    weather_df.index = weather_df.index.tz_convert(timezone)

    # shift date_time by 30 minutes, 'average time' during that hour
    # weather_df.index = weather_df.index.shift(freq="30min")

    return weather_df

# %%
# Whole year of 2023
start_date = '20230101'
end_date = '20231231'

timezone = 'Europe/Amsterdam'

# Eindhoven KNMI STATION
station = '370'
lat = 51.449772459909
lon = 5.3770039280214

# Function get_hourly_weather_data_for_pvlib defined in full code overview below
weather_df = get_hourly_weather_data_for_pvlib(station, start_date, end_date, timezone)

# %%
# Get solar position for the dates / times
solpos_df = get_solarposition(
    weather_df.index, latitude = lat,
    longitude = lon, altitude = 0,
    temperature = weather_df['T'])
solpos_df.index = weather_df.index

# Method 'Erbs' to go from GHI to DNI and DHI
irradiance_df = irradiance.erbs(weather_df['Q'], solpos_df['zenith'], weather_df.index)
irradiance_df['ghi'] = weather_df['Q']

# %%
# Add DNI and DHI to weather_df
columns = ['dni', 'dhi']
weather_df[columns] = irradiance_df[columns]

# Fill NaN values with 0
weather_df.fillna(0, inplace=True)

# %%
def read_and_process_csv(file_path):
    # Read the CSV file, with headers
    df = pd.read_csv(file_path)

    # Convert time to UTC standard
    df['time'] = pd.to_datetime(df['time'], utc=True)
        
    # Convert 'W.mean_value' to float
    df['W.mean_value'] = df['W.mean_value'].astype(str).str.replace('"', '').astype(float)
        
    return df

def merge_csv_files(file_pattern):
    # Find all CSV files matching the pattern
    files = glob(file_pattern)
    
    # Process each file and combine them
    df_list = [read_and_process_csv(file) for file in files]
    combined_df = pd.concat(df_list, ignore_index=True)
    
    # Sort by time
    combined_df.sort_values('time', inplace=True)

    # Fill empty values with 0
    combined_df.fillna(0, inplace=True)
    
    return combined_df

quarterly_output_df = merge_csv_files('C:/Users/20193362/Desktop/datadujuan/*.csv')

# %%
# From quarterly solar output data to hourly
hourly_output_df = quarterly_output_df[quarterly_output_df['time'].dt.minute == 0]
hourly_output_df = hourly_output_df.set_index('time')


# %%'
# Create completely merged dataset
merge = hourly_output_df.join(weather_df)

# Delete incomplete rows 
merge = merge.dropna()

# Save as a csv file
merge.to_csv('result.csv', sep=',', index=True, encoding='utf-8')
