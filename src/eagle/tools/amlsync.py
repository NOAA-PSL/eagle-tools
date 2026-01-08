import os
import logging
import tempfile
import json
import time

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import Metric

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

from eagle.tools.log import setup_simple_log

logger = logging.getLogger("eagle.tools")


def get_aml_tracking_uri():
    """Retrieves the Azure ML tracking URI from environment variables."""
    ml_client = MLClient(
        DefaultAzureCredential(),
        os.getenv("AZURE_SUBSCRIPTION_ID"),
        os.getenv("AZUREML_ARM_RESOURCEGROUP"),
        os.getenv("AZUREML_ARM_WORKSPACE_NAME"),
    )
    return ml_client.workspaces.get(ml_client.workspace_name).mlflow_tracking_uri


def log_params_by_category_as_artifacts(local_params_dict):
    """
    Separates parameters into 'config', 'metadata', and 'other' categories,
    then logs each category as a separate, named JSON artifact.

    Args:
        local_params_dict (dict): A dictionary of parameters to log.
    """
    if not local_params_dict:
        logger.info("  No parameters to log.")
        return

    logger.info("  Separating parameters into config, metadata, and other...")
    config_params = {}
    metadata_params = {}
    other_params = {}

    for key, value in local_params_dict.items():
        if key.startswith("config"):
            config_params[key] = value
        elif key.startswith("metadata"):
            metadata_params[key] = value
        else:
            other_params[key] = value

    # Helper function to log a dictionary directly as a named JSON artifact
    def _log_dict_as_json(params_dict, filename):
        if not params_dict:
            logger.info(f"  No '{filename}' parameters to log, skipping.")
            return

        logger.info(f"  Logging {len(params_dict)} parameters to '{filename}'...")
        # Use mlflow.log_dict to save the dictionary directly with the desired artifact name
        mlflow.log_dict(params_dict, f"parameters/{filename}")
        logger.info(f"    -> Successfully logged '{filename}'.")

    # Log each category to its own file
    _log_dict_as_json(config_params, "config_parameters.json")
    _log_dict_as_json(metadata_params, "metadata_parameters.json")
    _log_dict_as_json(other_params, "other_parameters.json")


def log_metrics_in_batches(local_client, remote_client, local_run_id, remote_run_id, local_metrics_dict):
    """
    Logs all metric history in batches to avoid hanging on many individual API calls.

    Args:
        local_client (MlflowClient): The client for the local MLflow instance.
        remote_client (MlflowClient): The client for the remote Azure ML instance.
        local_run_id (str): The ID of the local run to read from.
        remote_run_id (str): The ID of the remote run to write to.
        local_metrics_dict (dict): The dictionary of metrics from the local run.
    """
    METRIC_BATCH_SIZE = 100
    all_metrics_to_log = []

    logger.info("  Gathering all metric history points...")
    for metric_key in local_metrics_dict.keys():
        metric_history = local_client.get_metric_history(local_run_id, metric_key)
        for metric in metric_history:
            all_metrics_to_log.append(
                Metric(key=metric.key, value=metric.value, timestamp=metric.timestamp, step=metric.step)
            )

    if not all_metrics_to_log:
        logger.info("  No metrics to log.")
        return

    logger.info(f"  Found {len(all_metrics_to_log)} total metric points. Logging in batches of {METRIC_BATCH_SIZE}...")
    for i in range(0, len(all_metrics_to_log), METRIC_BATCH_SIZE):
        batch = all_metrics_to_log[i : i + METRIC_BATCH_SIZE]
        remote_client.log_batch(run_id=remote_run_id, metrics=batch)
        time.sleep(0.5)
        logger.info(f"    -> Logged a batch of {len(batch)} metric points.")


def main(offline_path, local_id, remote_name):
    """
    Note: Make sure these environment variables are set in your environment:
        AZURE_SUBSCRIPTION_ID, AZUREML_ARM_RESOURCEGROUP, AZUREML_ARM_WORKSPACE_NAME
    """

    setup_simple_log()
    offline_tracking_uri = f"file://{offline_path}"

    logger.info("Initializing clients...")
    local_client = MlflowClient(tracking_uri=offline_tracking_uri)

    logger.info("Searching for runs in local experiment...")
    runs_to_sync = local_client.search_runs(experiment_ids=[local_id])

    if not runs_to_sync:
        logger.info(f"No runs found in local experiment '{local_id}'. Exiting.")
        exit()

    logger.info(f"Found {len(runs_to_sync)} runs to sync.")

    logger.info("Setting up remote tracking URI and client for Azure ML...")
    aml_tracking_uri = get_aml_tracking_uri()
    mlflow.set_tracking_uri(aml_tracking_uri)
    remote_client = MlflowClient(tracking_uri=aml_tracking_uri)

    logger.info(f"Checking for remote experiment '{remote_name}'...")
    remote_experiment = mlflow.get_experiment_by_name(remote_name)
    if remote_experiment is None:
        remote_experiment_id = mlflow.create_experiment(remote_name)
        logger.info(f"Created new remote experiment with ID: {remote_experiment_id}")
    else:
        remote_experiment_id = remote_experiment.experiment_id
        logger.info(f"Found existing remote experiment with ID: {remote_experiment_id}")

    for run_info in runs_to_sync:
        local_run_id = run_info.info.run_id
        logger.info(f"Syncing local run: {local_run_id}")

        # Start the main parent run in Azure ML
        with mlflow.start_run(experiment_id=remote_experiment_id) as remote_run:
            remote_run_id = remote_run.info.run_id
            logger.info(f"Created main remote run: {remote_run_id}")

            local_run = local_client.get_run(local_run_id)

            # Log All Parameters as Separate JSON Artifacts
            log_params_by_category_as_artifacts(local_run.data.params)

            # Log Metrics to the Main Run in Batches
            logger.info("Logging metrics...")
            log_metrics_in_batches(local_client, remote_client, local_run_id, remote_run_id, local_run.data.metrics)

            # Log Artifacts to the Main Run
            logger.info("Logging other artifacts...")
            local_artifact_path = os.path.join(
                offline_path,
                local_id,
                local_run_id,
                "artifacts"
            )

            if os.path.exists(local_artifact_path) and os.listdir(local_artifact_path):
                mlflow.log_artifacts(local_artifact_path)
                logger.info("  Other artifacts logged.")
            else:
                logger.info("  No other artifacts found to log.")

    logger.info("Synchronization complete!")

