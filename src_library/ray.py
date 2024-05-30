
from ray.air.integrations.mlflow import MLflowLoggerCallback

run_config_with_callback = train.RunConfig(
    name="MLFlow logging experiment",
    callbacks=[MLflowLoggerCallback(
        tracking_uri="http://localhost:8080",  # Replace with your MLFlow tracking server URI.
        experiment_name="ray-tune-experiments",
        save_artifact=True
    )],
)

