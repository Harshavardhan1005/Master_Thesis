import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
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

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name1"])

    n_estimators = config["estimators"]["RandomForest"]["params"]["n_estimators"]
    max_depth = config["estimators"]["RandomForest"]["params"]["max_depth"]
    min_samples_split = config["estimators"]["RandomForest"]["params"]["min_samples_split"]
    min_samples_leaf = config["estimators"]["RandomForest"]["params"]["min_samples_leaf"]

    print('****************************************************************************************************')
    print('RandomForest Model')
    print('****************************************************************************************************')

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        lr = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            n_jobs=-1)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, mape) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        mlflow.log_param("min_samples_leaf", min_samples_leaf)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name=mlflow_config["registered_model_name1"])
        else:
            mlflow.sklearn.load_model(lr, "model")

        mlflow.end_run()

    mlflow.set_experiment(mlflow_config["experiment_name2"])
    n_estimators = config["estimators"]["XGBoost"]["params"]["n_estimators"]
    max_depth = config["estimators"]["XGBoost"]["params"]["max_depth"]
    min_child_weight = config["estimators"]["XGBoost"]["params"]["min_child_weight"]
    gamma = config["estimators"]["XGBoost"]["params"]["gamma"]
    learning_rate = config["estimators"]["XGBoost"]["params"]["learning_rate"]
    print('****************************************************************************************************')
    print('XGBoost Model')
    print('****************************************************************************************************')
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        lr = xgboost.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            gamma=gamma,
            learning_rate=learning_rate,
            verbosity=0,
            n_jobs=-1)

        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, mape) = eval_metrics(test_y, predicted_qualities)

        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_child_weight", min_child_weight)
        mlflow.log_param("gamma", gamma)
        mlflow.log_param("learning_rate", learning_rate)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                lr,
                "model",
                registered_model_name=mlflow_config["registered_model_name2"])
        else:
            mlflow.sklearn.load_model(lr, "model")

        mlflow.end_run()

    print('****************************************************************************************************')
    print('LSTM Model')
    print('****************************************************************************************************')

    mlflow.set_experiment(mlflow_config["experiment_name3"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        # Scaling the data
        scaler = MinMaxScaler()
        scaler.fit(train_x)
        scaled_train = scaler.transform(train_x)
        scaled_test = scaler.transform(test_x)

        # Train generator
        length = config["estimators"]["LSTM"]["params"]["length"]
        epochs = config["estimators"]["LSTM"]["params"]["epoch"]

        train_generator = TimeseriesGenerator(scaled_train, train_y.values, length=length, batch_size=1)

        model = Sequential()
        model.add(LSTM(50, activation='relu', return_sequences=True, input_shape=(length, 14)))
        model.add(Dropout(0.2))
        model.add(LSTM(50, activation='relu'))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mse')

        model.fit(train_generator, epochs=epochs)

        # Test data predictions
        test_predictions = []

        # last n_input points from the training set
        first_eval_batch = scaled_train[-length:]
        # reshape this to the format of RNN (same format as TimeseriesGeneration)
        current_batch = first_eval_batch.reshape((1, length, 14))

        for i in range(len(test_x)):
            # One timestep ahead of historical 12 points
            current_pred = model.predict(current_batch)[0]
            # store that prediction
            test_predictions.append(current_pred)

            # update the current batch to include prediction
            current_batch = np.append(current_batch[:, 1:, :], [[scaled_test[i]]], axis=1)

        (rmse, mae, mape) = eval_metrics(test_y, test_predictions)

        mlflow.log_param("length", length)
        mlflow.log_param("epochs", epochs)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=mlflow_config["registered_model_name3"])

        else:
            mlflow.sklearn.load_model(model, "model")

        mlflow.end_run()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
