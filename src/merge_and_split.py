import os

import pandas as pd
import argparse
import yaml
from sklearn.model_selection import train_test_split


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config



def merge_and_split_data(config_path):
    config = read_params(config_path)
    gps_data_path = config["load_data"]["gps_dataset_csv"]
    weather_data_path = config["load_data"]["weather_dataset_csv"]
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    random_state = config["base"]["random_state"]
    split_ratio = config["split_data"]["test_size"]
    data_merge = config["data_source"]["data_merge"]

    df_gps = pd.read_csv(gps_data_path)
    df_weather = pd.read_csv(weather_data_path)

    df_weather['Timestamp'] = pd.to_datetime(df_weather['Timestamp'])
    df_gps['start_plant2'] = pd.to_datetime(df_gps['start_plant2'])

    df_weather['time'] = df_weather['Timestamp'].apply(lambda x: x.strftime("%Y-%m-%d %H"))
    df_gps['time'] = df_gps['start_plant2'].apply(lambda x: x.strftime("%Y-%m-%d %H"))

    merge_df = pd.merge(df_gps,df_weather,on='time')

    merge_df = merge_df[['Week_Day','Week','Hour','Minutes','Seconds','speed_threshold','Clouds','Temp','Wind_deg','Wind_speed','Rain_1h','Rain_3h','Snow_1h','Snow_3h','travel_time(2-1)']]

    train, test = train_test_split(
        merge_df,
        test_size=split_ratio,
        random_state=random_state,
        shuffle=False
    )

    os.makedirs(data_merge,exist_ok=True)
    train.to_csv(train_data_path, sep=",", index=False, encoding="utf-8")
    test.to_csv(test_data_path, sep=",", index=False, encoding="utf-8")

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    merge_and_split_data(config_path=parsed_args.config)