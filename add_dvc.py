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

# Do not forget to add weather and fms data too

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    gps_raw_data(config_path=parsed_args.config)

