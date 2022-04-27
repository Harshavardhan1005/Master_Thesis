#For getting the weather data from API, uncomment the 'Store data' in the function 'clean_weather_data', this should be uncomment only once and later times, comment the funciton(line) in order to duplicate the data or to reduce the API call's

# Import all the necessary libraries
import os
import yaml
import pandas as pd
import math
import numpy as np
from ast import literal_eval
from tqdm import tqdm
import argparse
tqdm.pandas(desc="Progress!")


# Function to read the configuration file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


# Function to get the weather data
def store_weather_data(config_path):
    config = read_params(config_path)
    start = config["data_source"]["weather_start"]
    end = config["data_source"]["weather_end"]
    weather_raw_data_source_path = config["data_source"]["weather_raw_data_source"]
    weather_raw_data_path = config["data_source"]["weather_raw_data"]

    df_weather = pd.DataFrame()

    for i in tqdm(range(math.ceil((end - start) / (167 * 3600)))):
        df = pd.read_json(
            'http://history.openweathermap.org/data/2.5/history/city?q=Friedrichshafen,DE&type=hour&start={0}&cnt=168&appid=212117db1236e6aee483f90d1592f01b'.format(
                start))
        df_weather = pd.concat([df_weather, df])
        start = start + (167 * 3600)

    os.makedirs(weather_raw_data_path, exist_ok=True)
    df_weather.to_csv(weather_raw_data_source_path, index=False)

    
# Function to clean the weather data
def clean_weather_data(config_path):
    config = read_params(config_path)
    #store_weather_data(config_path=config_path)
    weather_cleaned_data_source_path = config["data_source"]["weather_cleaned_data_source"]
    weather_cleaned_data_path = config["data_source"]["weather_cleaned_data"]
    weather_raw_data_source_path = config["data_source"]["weather_raw_data_source"]

    Timestamp = []
    Clouds = []
    Temp = []
    Weather = []
    Wind_deg = []
    Wind_speed = []
    Rain_1h = []
    Rain_3h = []
    Snow_1h = []
    Snow_3h = []

    df_weather = pd.read_csv(weather_raw_data_source_path)
    df_weather['list'] = df_weather['list'].apply(lambda x: literal_eval(x))
    final_weather = pd.DataFrame(
        columns=['Timestamp', 'Clouds', 'Temp', 'Weather', 'Wind_deg', 'Wind_speed', 'Rain_1h', 'Rain_3h', 'Snow_1h',
                 'Snow_3h'])

    for row in df_weather.to_dict('records'):
        Timestamp.append(pd.to_datetime(row['list']['dt'], unit='s').tz_localize('utc').tz_convert('Europe/Berlin'))
        Clouds.append(row['list']['clouds']['all'])
        Temp.append(row['list']['main']['temp'])
        Weather.append(row['list']['weather'][0]['description'])

        if 'rain' in row['list'].keys():
            if '1h' in row['list']['rain'].keys():
                Rain_1h.append(row['list']['rain']['1h'])
            else:
                Rain_1h.append(np.NaN)
            if '3h' in row['list']['rain'].keys():
                Rain_3h.append(row['list']['rain']['3h'])
            else:
                Rain_3h.append(np.NaN)
        else:
            Rain_1h.append(np.NaN)
            Rain_3h.append(np.NaN)

        if 'snow' in row['list'].keys():
            if '1h' in row['list']['snow'].keys():
                Snow_1h.append(row['list']['snow']['1h'])
            else:
                Snow_1h.append(np.NaN)
            if '3h' in row['list']['snow'].keys():
                Snow_3h.append(row['list']['snow']['3h'])
            else:
                Snow_3h.append(np.NaN)
        else:
            Snow_1h.append(np.NaN)
            Snow_3h.append(np.NaN)

        if 'wind' in row['list'].keys():
            Wind_deg.append(row['list']['wind']['deg'])
            Wind_speed.append(row['list']['wind']['speed'])
        else:
            Wind_deg.append(np.NaN)
            Wind_speed.append(np.NaN)

    final_weather['Timestamp'] = Timestamp
    final_weather['Clouds'] = Clouds
    final_weather['Temp'] = Temp
    final_weather['Weather'] = Weather
    final_weather['Rain_1h'] = Rain_1h
    final_weather['Rain_3h'] = Rain_3h
    final_weather['Snow_1h'] = Snow_1h
    final_weather['Snow_3h'] = Snow_3h
    final_weather['Wind_deg'] = Wind_deg
    final_weather['Wind_speed'] = Wind_speed

    os.makedirs(weather_cleaned_data_path, exist_ok=True)
    final_weather.to_csv(weather_cleaned_data_source_path, index=False)

    
# Function to preprocess the weather data
def preprocess_weather_data(config_path):
    clean_weather_data(config_path)
    config = read_params(config_path)
    weather_cleaned_data_source_path = config["data_source"]["weather_cleaned_data_source"]
    weather_dataset_csv = config["load_data"]["weather_dataset_csv"]
    weather_preprocessed_data_path = config["data_source"]["weather_preprocessed_data"]
    df = pd.read_csv(weather_cleaned_data_source_path)
    df.fillna(0, inplace=True)
    os.makedirs(weather_preprocessed_data_path, exist_ok=True)
    df.to_csv(weather_dataset_csv, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    preprocess_weather_data(config_path=parsed_args.config)
