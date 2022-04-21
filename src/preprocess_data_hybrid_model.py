# Import all the necessary libraries
import os
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import argparse
import joblib
import yaml
import logging
import warnings
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

# Function to read the configuration file
def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

# Function to preprocess the data required by the hybrid model
def preprocess_data_hybrid_model(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    hybrid_train_path = config["split_data"]["hybrid_train_path"]
    hybrid_test_path = config["split_data"]["hybrid_test_path"]
    target = [config["base"]["target_col"]]
    rf_model_dir_path = config["rf_model_dir"]
    xgb_model_dir_path = config["xgb_model_dir"]
    lstm_model_dir_path = config["lstm_model_dir"]

    rf_model = joblib.load(rf_model_dir_path)
    xgb_model = joblib.load(xgb_model_dir_path)
    lstm_model = joblib.load(lstm_model_dir_path)

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    # Scaling the data
    scaler = MinMaxScaler()
    scaler.fit(train_x)
    scaled_train = scaler.transform(train_x)
    scaled_test = scaler.transform(test_x)

    train_df = pd.DataFrame(columns=['rf_model', 'xgb_model', 'lstm_model', 'travel_time'])
    test_df = pd.DataFrame(columns=['rf_model', 'xgb_model', 'lstm_model', 'travel_time'])
    length = 1
    train_predictions = []
    first_eval_batch = scaled_train[0:length]
    current_batch = first_eval_batch.reshape((1, length, 14))
    for i in range(len(train_x[length:])):
        current_pred = lstm_model.predict(current_batch)[0][0]
        train_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[scaled_train[i]]], axis=1)

    test_predictions = []
    first_eval_batch = scaled_train[-length:]
    current_batch = first_eval_batch.reshape((1, length, 14))
    for i in range(len(test_x)):
        current_pred = lstm_model.predict(current_batch)[0][0]
        test_predictions.append(current_pred)
        current_batch = np.append(current_batch[:, 1:, :], [[scaled_test[i]]], axis=1)

    train_df['rf_model'] = rf_model.predict(train_x)[length:]
    train_df['xgb_model'] = xgb_model.predict(train_x)[length:]
    test_df['rf_model'] = rf_model.predict(test_x)
    test_df['xgb_model'] = xgb_model.predict(test_x)
    train_df['lstm_model'] = train_predictions
    test_df['lstm_model'] = test_predictions
    train_df['travel_time'] = train_y[length:].reset_index(drop=True)
    test_df['travel_time'] = test_y

    train_df.to_csv(hybrid_train_path,index=False)
    test_df.to_csv(hybrid_test_path,index=False)

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    preprocess_data_hybrid_model(config_path=parsed_args.config)
