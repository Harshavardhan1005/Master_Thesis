import argparse
import mlflow
import pandas as pd
from mlflow.tracking import MlflowClient
from pprint import pprint
import joblib
import yaml
import warnings

warnings.filterwarnings('ignore')


def read_params(config_path):
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def log_final_model(config_path):
    config = read_params(config_path)
    df = pd.DataFrame(columns=['Model_Source', 'Score'])

    mlflow_config = config["mlflow_config"]
    model_name1 = mlflow_config["registered_model_name1"]
    model_name2 = mlflow_config["registered_model_name2"]
    model_name3 = mlflow_config["registered_model_name3"]

    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)

    runs = mlflow.search_runs(experiment_ids='1')
    lowest = list(runs["metrics.mape"].sort_values(ascending=True))[0]
    lowest_run_id = runs[runs["metrics.mape"] == lowest]["run_id"]

    client = MlflowClient()
    for mv in client.search_model_versions(f"name='{model_name1}'"):
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id.iloc[0]:
            current_version = mv["version"]
            logged_model1 = mv["source"]

            df.loc[0, 'Model_Source'] = logged_model1
            df.loc[0, 'Score'] = lowest

            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name1,
                version=current_version,
                stage="Production"
            )
            rf_model_path = config["rf_model_dir"]
            model = mlflow.pyfunc.load_model(logged_model1)
            joblib.dump(model, rf_model_path)

        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name1,
                version=current_version,
                stage="Staging"
            )

    runs = mlflow.search_runs(experiment_ids='2')
    lowest = list(runs["metrics.mape"].sort_values(ascending=True))[0]
    lowest_run_id = runs[runs["metrics.mape"] == lowest]["run_id"]

    for mv in client.search_model_versions(f"name='{model_name2}'"):
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id.iloc[0]:
            current_version = mv["version"]
            logged_model2 = mv["source"]
            df.loc[1, 'Model_Source'] = logged_model2
            df.loc[1, 'Score'] = lowest
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name2,
                version=current_version,
                stage="Production"
            )
            xgb_model_path = config["xgb_model_dir"]
            model = mlflow.pyfunc.load_model(logged_model2)
            joblib.dump(model, xgb_model_path)
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name2,
                version=current_version,
                stage="Staging"
            )

    runs = mlflow.search_runs(experiment_ids='3')
    lowest = list(runs["metrics.mape"].sort_values(ascending=True))[0]
    lowest_run_id = runs[runs["metrics.mape"] == lowest]["run_id"]

    for mv in client.search_model_versions(f"name='{model_name3}'"):
        mv = dict(mv)
        if mv["run_id"] == lowest_run_id.iloc[0]:
            current_version = mv["version"]
            logged_model3 = mv["source"]
            df.loc[2, 'Model_Source'] = logged_model3
            df.loc[2, 'Score'] = lowest
            pprint(mv, indent=4)
            client.transition_model_version_stage(
                name=model_name3,
                version=current_version,
                stage="Production"
            )
            lstm_model_path = config["lstm_model_dir"]
            model = mlflow.pyfunc.load_model(logged_model3)
            joblib.dump(model, lstm_model_path)
        else:
            current_version = mv["version"]
            client.transition_model_version_stage(
                name=model_name3,
                version=current_version,
                stage="Staging"
            )

    best_model = df.iloc[df['Score'][0:-1].astype(float).idxmin()]['Model_Source']

    loaded_model = mlflow.pyfunc.load_model(best_model)
    model_path = config["webapp_model_dir"]
    joblib.dump(loaded_model, model_path)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    log_final_model(config_path=parsed_args.config)
