import os
import yaml
import pandas as pd
import numpy as np
from glob import glob
from tqdm import tqdm
import argparse

tqdm.pandas(desc="Progress!")


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def load_fms_data(config_path):
    config = read_params(config_path)

    fms_data_path = config["data_source"]["fms_data_source"]
    fms_data_cols = config["data_source"]["fms_data_cols"]
    df_fms = pd.read_csv(fms_data_path, sep=';', usecols=fms_data_cols, encoding='utf-8')
    df_fms.rename(columns={'Code': 'key'}, inplace=True)
    df_fms['key'] = df_fms['key'].astype(str)

    return df_fms


def load_and_clean_gps_data(config_path):
    config = read_params(config_path)

    gps_raw_data_path = config["data_source"]["gps_raw_data"]
    gps_raw_data_cols = config["data_source"]["gps_raw_data_cols"]
    gps_cleaned_data_path = config["data_source"]["gps_cleaned_data"]
    df_fms = load_fms_data(config_path)

    for file in tqdm(glob(gps_raw_data_path + '/*'),desc='Loading and Cleaning gps Data'):
        final_df = pd.DataFrame()
        for folder in np.sort(glob(file + '/*')):
            for path in glob(folder + '/*.csv.gz'):

                df_raw = pd.read_csv(path,compression='gzip',sep='|', usecols=gps_raw_data_cols,low_memory=False)
                df_raw['Time_stamp'] = pd.to_datetime(df_raw['ts_msg_usec'] + df_raw['timedelta_usec'], unit='us')
                df_raw['Time_stamp'] = df_raw['Time_stamp'].astype('datetime64[s]').dt.tz_localize('utc').dt.tz_convert(
                    'Europe/Berlin')
                df_raw.drop_duplicates(subset=['Time_stamp', 'key'], keep='last', inplace=True)

                merge_df = df_raw.merge(df_fms, on='key', how='left')
                merge_df.Name.fillna(merge_df.key, inplace=True)
                merge_df.value2.fillna(merge_df.value, inplace=True)

                df_raw = merge_df.pivot(index='Time_stamp', columns='Name', values='value2')
                df_raw.reset_index(inplace=True)
                df_raw['lat'] = df_raw['lat'].astype(float)
                df_raw['lon'] = df_raw['lon'].astype(float)
                df_raw.columns.name = None

                df1 = pd.DataFrame()
                df1['Time_stamp'] = pd.date_range(df_raw['Time_stamp'][0], df_raw['Time_stamp'][len(df_raw) - 1],
                                                  freq='1s')
                df1 = df1.merge(df_raw, on='Time_stamp', how='left')
                df1['lat'] = df1['lat'].interpolate().ffill().bfill()
                df1['lon'] = df1['lon'].interpolate().ffill().bfill()
                df1['WheelBasedVehicleSpeed'] = df1['WheelBasedVehicleSpeed'].interpolate().ffill().bfill()
                final_df = pd.concat([final_df,df1])

        os.makedirs(gps_cleaned_data_path,exist_ok=True)
        final_df.to_csv(gps_cleaned_data_path + '/' + str(file[-7:]) + '.csv', sep=",", date_format='%Y/%m/%d %H:%M:%S', index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    load_and_clean_gps_data(config_path=parsed_args.config)
