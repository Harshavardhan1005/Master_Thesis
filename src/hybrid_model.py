import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import argparse
import mlflow
import yaml
import logging
from urllib.parse import urlparse

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


def train_hybrid_model(config_path):
    config = read_params(config_path)
    target = [config["base"]["target_col"]]

    hybrid_train_path = config["split_data"]["hybrid_train_path"]
    hybrid_test_path = config["split_data"]["hybrid_test_path"]

    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]
    mlflow.set_tracking_uri(remote_server_uri)

    train = pd.read_csv(hybrid_train_path, sep=",")
    test = pd.read_csv(hybrid_test_path, sep=",")

    train_y = train[target]
    test_y = test[target]

    train_x = train.drop(target, axis=1)
    test_x = test.drop(target, axis=1)

    mlflow.set_experiment(mlflow_config["experiment_name4"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        reg = LinearRegression()
        reg.fit(train_x[['rf_model','lstm_model']],train_y)

        predictions = reg.predict(test_x[['rf_model','lstm_model']])
        (rmse, mae, mape) = eval_metrics(test_y, predictions)

        mlflow.log_param("coefficient", reg.coef_)
        mlflow.log_param("intercept", reg.intercept_)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                reg,
                "model",
                registered_model_name=mlflow_config["registered_model_name4"])
        else:
            mlflow.sklearn.load_model(reg, "model")

        mlflow.end_run()

    mlflow.set_experiment(mlflow_config["experiment_name5"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        reg = LinearRegression()
        reg.fit(train_x[['xgb_model', 'lstm_model']], train_y)

        predictions = reg.predict(test_x[['xgb_model', 'lstm_model']])
        (rmse, mae, mape) = eval_metrics(test_y, predictions)

        mlflow.log_param("coefficient", reg.coef_)
        mlflow.log_param("intercept", reg.intercept_)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                reg,
                "model",
                registered_model_name=mlflow_config["registered_model_name5"])
        else:
            mlflow.sklearn.load_model(reg, "model")

        mlflow.end_run()

    mlflow.set_experiment(mlflow_config["experiment_name6"])
    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:

        reg = LinearRegression()
        reg.fit(train_x[['rf_model', 'xgb_model']], train_y)

        predictions = reg.predict(test_x[['rf_model', 'xgb_model']])
        (rmse, mae, mape) = eval_metrics(test_y, predictions)

        mlflow.log_param("coefficient", reg.coef_)
        mlflow.log_param("intercept", reg.intercept_)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mape", mape)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                reg,
                "model",
                registered_model_name=mlflow_config["registered_model_name6"])
        else:
            mlflow.sklearn.load_model(reg, "model")

        mlflow.end_run()

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_hybrid_model(config_path=parsed_args.config)
