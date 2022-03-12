import yaml
import argparse
from glob import glob
import numpy as np
import os


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def gps_raw_data(config_path):
    config = read_params(config_path)
    gps_raw_data_path = config["data_source"]["gps_raw_data"]
    for file in np.sort(glob(gps_raw_data_path + '/*')):
        for folder in np.sort(glob(file + '/*')):
            for path in glob(folder + '/*.csv.gz'):
                os.system(f"dvc add {path}")


def gps_cleaned_data(config_path):
    config = read_params(config_path)
    gps_cleaned_data_path = config["data_source"]["gps_cleaned_data"]
    for path in np.sort(glob(gps_cleaned_data_path + '/*.csv')):
        os.system(f"dvc add {path}")


def gps_preprocessed_data(config_path):
    config = read_params(config_path)
    gps_preprocessed_data1_path = config["data_source"]["gps_preprocessed_data1"]
    gps_preprocessed_data2_path = config["data_source"]["gps_preprocessed_data2"]
    for path in glob(gps_preprocessed_data1_path + '/*.csv'):
        os.system(f"dvc add {path}")
    for path in glob(gps_preprocessed_data2_path + '/*.csv'):
        os.system(f"dvc add {path}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    gps_raw_data(config_path=parsed_args.config)
    #gps_cleaned_data(config_path=parsed_args.config)
    #gps_preprocessed_data(config_path=parsed_args.config)
