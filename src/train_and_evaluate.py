import os

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import argparse
import itertools
import xgboost
import mlflow
import yaml
import logging
from urllib.parse import urlparse
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.preprocessing.sequence import TimeseriesGenerator
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


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
    random_state = config["base"]["random_state"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)
    '''
    print('****************************************************************************************************')
    print('RandomForest Model')
    print('****************************************************************************************************')

    mlflow.set_experiment(mlflow_config["experiment_name1"])

    n_estimators = config["estimators"]["RandomForest"]["params"]["n_estimators"]
    max_depth = config["estimators"]["RandomForest"]["params"]["max_depth"]
    min_samples_split = config["estimators"]["RandomForest"]["params"]["min_samples_split"]
    min_samples_leaf = config["estimators"]["RandomForest"]["params"]["min_samples_leaf"]
    rf_params_accuracy = config["estimators"]['RandomForest']["rf_params_accuracy"]

    rf_params_accuracy_csv = pd.DataFrame(
        columns=['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'rmse', 'mae', 'mape'])

    for x in tqdm(list(itertools.product(n_estimators, max_depth, min_samples_split, min_samples_leaf))):

        with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

            rf = RandomForestRegressor(
                n_estimators=x[0],
                max_depth=x[1],
                min_samples_split=x[2],
                min_samples_leaf=x[3],
                random_state=random_state,
                n_jobs=-1)
            rf.fit(train_x, train_y)

            predictions = rf.predict(test_x)

            (rmse, mae, mape) = eval_metrics(test_y, predictions)

            mlflow.log_param("n_estimators", x[0])
            mlflow.log_param("max_depth", x[1])
            mlflow.log_param("min_samples_split", x[2])
            mlflow.log_param("min_samples_leaf", x[3])

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    rf,
                    "model",
                    registered_model_name=mlflow_config["registered_model_name1"])
            else:
                mlflow.sklearn.load_model(rf, "model")

            mlflow.end_run()

            rf_params_accuracy_one = pd.DataFrame(index=range(1),
                                                  columns=['n_estimators', 'max_depth', 'min_samples_split',
                                                           'min_samples_leaf', 'rmse', 'mae', 'mape'])

            rf_params_accuracy_one.loc[:, 'n_estimators'] = x[0]
            rf_params_accuracy_one.loc[:, 'max_depth'] = x[1]
            rf_params_accuracy_one.loc[:, 'min_samples_split'] = x[2]
            rf_params_accuracy_one.loc[:, 'min_samples_leaf'] = x[3]
            rf_params_accuracy_one.loc[:, 'rmse'] = rmse
            rf_params_accuracy_one.loc[:, 'mae'] = mae
            rf_params_accuracy_one.loc[:, 'mape'] = mape
            rf_params_accuracy_csv = rf_params_accuracy_csv.append(rf_params_accuracy_one)

    rf_params_accuracy_csv.reset_index(drop=True, inplace=True)
    rf_params_accuracy_csv.to_csv(rf_params_accuracy, index=False)
    '''
    print('****************************************************************************************************')
    print('XGBoost Model')
    print('****************************************************************************************************')

    mlflow.set_experiment(mlflow_config["experiment_name2"])

    n_estimators = config["estimators"]["XGBoost"]["params"]["n_estimators"]
    max_depth = config["estimators"]["XGBoost"]["params"]["max_depth"]
    min_child_weight = config["estimators"]["XGBoost"]["params"]["min_child_weight"]
    gamma = config["estimators"]["XGBoost"]["params"]["gamma"]
    learning_rate = config["estimators"]["XGBoost"]["params"]["learning_rate"]
    xgb_params_accuracy = config["estimators"]['XGBoost']["xgb_params_accuracy"]

    xgb_params_accuracy_csv = pd.DataFrame(
        columns=['n_estimators', 'max_depth', 'min_child_weight', 'gamma', 'learning_rate', 'rmse', 'mae', 'mape'])

    for x in tqdm(list(itertools.product(n_estimators, max_depth, min_child_weight, gamma, learning_rate))):

        with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

            xgb = xgboost.XGBRegressor(
                    n_estimators=x[0],
                    max_depth=x[1],
                    min_child_weight=x[2],
                    gamma=x[3],
                    learning_rate=x[4],
                    verbosity=0,
                    n_jobs=-1)

            xgb.fit(train_x,train_y)

            predictions = xgb.predict(test_x)

            (rmse, mae, mape) = eval_metrics(test_y, predictions)

            mlflow.log_param("n_estimators", x[0])
            mlflow.log_param("max_depth", x[1])
            mlflow.log_param("min_child_weight", x[2])
            mlflow.log_param("gamma", x[3])
            mlflow.log_param("learning_rate", x[4])

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    xgb,
                    "model",
                    registered_model_name=mlflow_config["registered_model_name2"])
            else:
                mlflow.sklearn.load_model(xgb, "model")

            mlflow.end_run()

            xgb_params_accuracy_one = pd.DataFrame(index=range(1),
                                                  columns=['n_estimators', 'max_depth', 'min_child_weight', 'gamma',
                                                           'learning_rate', 'rmse', 'mae', 'mape'])

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


    print('****************************************************************************************************')
    print('LSTM Model')
    print('****************************************************************************************************')

    mlflow.set_experiment(mlflow_config["experiment_name3"])

    length = config["estimators"]["LSTM"]["params"]["length"]
    epochs = config["estimators"]["LSTM"]["params"]["epoch"]
    lstm_params_accuracy = config["estimators"]['LSTM']["lstm_params_accuracy"]

    lstm_params_accuracy_csv = pd.DataFrame(
        columns=['length', 'epochs', 'rmse', 'mae', 'mape'])

    # Scaling the data
    scaler = MinMaxScaler()
    scaler.fit(train_x)
    scaled_train = scaler.transform(train_x)
    scaled_test = scaler.transform(test_x)

    for x in tqdm(list(itertools.product(length, epochs))):

        with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

            train_generator = TimeseriesGenerator(scaled_train, train_y.values, length=x[0], batch_size=1)

            model = Sequential()
            model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(x[0], 14)))
            model.add(Dropout(0.2))
            model.add(LSTM(50, activation='relu'))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mse')

            model.fit(train_generator, epochs=x[1])

            # Test data predictions
            test_predictions = []

            # last n_input points from the training set
            first_eval_batch = scaled_train[-x[0]:]
            # reshape this to the format of RNN (same format as TimeseriesGeneration)
            current_batch = first_eval_batch.reshape((1, x[0], 14))

            for i in range(len(test_x)):
                # One timestep ahead of historical 12 points
                current_pred = model.predict(current_batch)[0]
                # store that prediction
                test_predictions.append(current_pred)

                # update the current batch to include prediction
                current_batch = np.append(current_batch[:, 1:, :], [[scaled_test[i]]], axis=1)

            (rmse, mae, mape) = eval_metrics(test_y, test_predictions)

            mlflow.log_param("length", x[0])
            mlflow.log_param("epochs", x[1])

            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("mape", mape)

            tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

            if tracking_url_type_store != "file":
                mlflow.sklearn.log_model(
                    model,
                    "model",
                    registered_model_name=mlflow_config["registered_model_name2"])
            else:
                mlflow.sklearn.load_model(xgb, "model")

            mlflow.end_run()

            lstm_params_accuracy_one = pd.DataFrame(index=range(1),
                                                  columns=['length', 'epochs','rmse', 'mae', 'mape'])

            lstm_params_accuracy_one.loc[:, 'n_estimators'] = x[0]
            lstm_params_accuracy_one.loc[:, 'max_depth'] = x[1]
            lstm_params_accuracy_one.loc[:, 'rmse'] = rmse
            lstm_params_accuracy_one.loc[:, 'mae'] = mae
            lstm_params_accuracy_one.loc[:, 'mape'] = mape
            lstm_params_accuracy_csv = lstm_params_accuracy_csv.append(lstm_params_accuracy_one)

    lstm_params_accuracy_csv.reset_index(drop=True, inplace=True)
    lstm_params_accuracy_csv.to_csv(lstm_params_accuracy, index=False)


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
