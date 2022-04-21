# Import all the necessary libraries
import os
import yaml
import pandas as pd
from glob import glob
import argparse

# Function to read the config file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

# Function to merge the GPS data 
def merge_gps_data(config_path):
    config = read_params(config_path)
    gps_preprocessed_data2 = config["data_source"]["gps_preprocessed_data2"]
    gps_preprocessed_data1 = config["data_source"]["gps_preprocessed_data1"]

    gps_merge_data_path = config["data_source"]["gps_merge_data"]
    gps_dataset_csv = config["load_data"]["gps_dataset_csv"]

    df_gps = pd.DataFrame()
    for path in glob(gps_preprocessed_data2 + '/*'):
        df = pd.read_csv(path)
        df_gps = pd.concat([df, df_gps])


    ####### 
    for path in glob(gps_preprocessed_data1 + '/*'):
        df = pd.read_csv(path)
        df.rename(columns={"start_plant1": "start_plant2", "end_plant2": "end_plant1", "route_1_2":"route_2_1"},
                  inplace=True)
        df_gps = pd.concat([df, df_gps])
    ######


    df_gps = df_gps[df_gps['route_2_1'] == 1]
    df_gps = df_gps[df_gps['travel_time'] < 20]

    df_gps['start_plant2'] = pd.to_datetime(df_gps['start_plant2'])
    df_gps['Week_Day_Name'] = df_gps['start_plant2'].dt.day_name()
    df_gps['Week_Day'] = df_gps['start_plant2'].dt.weekday
    df_gps['Week'] = df_gps['start_plant2'].dt.isocalendar().week
    df_gps['time'] = df_gps['start_plant2'].dt.time
    df_gps['Hour'] = df_gps['time'].apply(lambda x: x.hour)
    df_gps['Minutes'] = df_gps['time'].apply(lambda x: x.minute)
    df_gps['Seconds'] = df_gps['time'].apply(lambda x: x.second)

    df_gps.sort_values('start_plant2',inplace=True)
    os.makedirs(gps_merge_data_path, exist_ok=True)
    df_gps.to_csv(gps_dataset_csv, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    merge_gps_data(config_path=parsed_args.config)
