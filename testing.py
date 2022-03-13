import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import argparse
import xgboost
import mlflow
import yaml
import logging
from urllib.parse import urlparse
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
import warnings
warnings.filterwarnings('ignore')
import itertools

def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    mape = mean_absolute_percentage_error(actual, pred)
    return rmse, mae, mape

def train_and_evaluate(config_path):
    config = read_params(config_path)
    test_data_path = config["split_data"]["test_path"]
    train_data_path = config["split_data"]["train_path"]
    target = [config["base"]["target_col"]]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    print(train_x)
    print(train_y)
    n_estimators = config["estimators"]["XGBoost"]["params"]["n_estimators"]
    max_depth = config["estimators"]["XGBoost"]["params"]["max_depth"]
    min_child_weight = config["estimators"]["XGBoost"]["params"]["min_child_weight"]
    gamma = config["estimators"]["XGBoost"]["params"]["gamma"]
    learning_rate = config["estimators"]["XGBoost"]["params"]["learning_rate"]
    xgb_params_accuracy = config["estimators"]['XGBoost']["xgb_params_accuracy"]

    xgb_params_accuracy_csv = pd.DataFrame(
        columns=['n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'learning_rate', 'rmse', 'mae', 'mape'])

    for x in tqdm(list(itertools.product(n_estimators, max_depth, min_child_weight, gamma, learning_rate))):
        xgb = xgboost.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            learning_rate=learning_rate,
            verbosity=0,
            n_jobs=-1)

        xgb.fit(train_x, train_y)

        predictions = xgb.predict(test_x)

        (rmse, mae, mape) = eval_metrics(test_y, predictions)

        xgb_params_accuracy_one = pd.DataFrame(index=range(1),
                                                  columns=['n_estimators', 'max_depth', 'min_child_weight', 'gamma',
                                                           'learning_rate', 'rmse', 'mae', 'mape' 'rmse', 'mae', 'mape'])

        xgb_params_accuracy_one.loc[:, 'n_estimators'] = x[0]
        xgb_params_accuracy_one.loc[:, 'max_depth'] = x[1]
        xgb_params_accuracy_one.loc[:, 'min_child_weight'] = x[2]
        xgb_params_accuracy_one.loc[:, 'gamma'] = x[3]
        xgb_params_accuracy_one.loc[:, 'learning_rate'] = x[4]
        xgb_params_accuracy_one.loc[:, 'rmse'] = rmse
        xgb_params_accuracy_one.loc[:, 'mae'] = mae
        xgb_params_accuracy_one.loc[:, 'mape'] = mape
        xgb_params_accuracy_csv = xgb_params_accuracy_csv.append(xgb_params_accuracy_one)

    xgb_params_accuracy_csv.reset_index(drop=True, inplace=True)
    xgb_params_accuracy_csv.to_csv(xgb_params_accuracy, index=False)



if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)