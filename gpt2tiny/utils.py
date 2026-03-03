from typing import Literal
from pathlib import Path
import mlflow


def _download_resume_ckpt(
    *,
    experiment_name: str,
    run_name: str,
    which: Literal["last", "best"],
    dst_dir: Path,
    tracking_uri: str,
) -> Path:
    """
    Finds an MLflow run by run_name (tag mlflow.runName) and downloads either:
      - checkpoints/last.ckpt
      - checkpoints/best.ckpt
    Returns the local path.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")

    # MLflow stores run name in tag "mlflow.runName"
    runs = client.search_runs(
        experiment_ids=[exp.experiment_id],
        filter_string=f"tag.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=5,
    )
    if not runs:
        raise RuntimeError(f"No run found with name='{run_name}' in experiment='{experiment_name}'")

    run = runs[0]
    run_id = run.info.run_id

    artifact_rel_path = f"checkpoints/{which}.ckpt"

    dst_dir.mkdir(parents=True, exist_ok=True)
    local_path = client.download_artifacts(run_id, artifact_rel_path, dst_path=str(dst_dir))
    return Path(local_path)