"""Model loading functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow.artifacts import download_artifacts
from mlflow.models import Model

from mlflow_autogluon.constants import AUTODEPLOY_SUBPATH, FLAVOR_NAME
from mlflow_autogluon.literals import ModelTypeLiteral


def load_model(
    model_uri: str,
    dst_path: str | None = None,
) -> Any | object:  # noqa: WPS210,WPS211
    """
    Load an AutoGluon model from MLflow.

    Args:
        model_uri: URI pointing to the model (e.g., 'runs:/<run_id>/model')
        dst_path: Optional local path to download model to

    Returns:
        Loaded AutoGluon model instance

    Raises:
        ValueError: If model cannot be loaded or flavor configuration is invalid
    """
    # If model_uri is a local directory, load directly without
    # calling MLflow download_artifacts()
    p = Path(model_uri)
    if p.exists() and p.is_dir():
        return _load_model_from_local_path(p)

    local_model_path = download_artifacts(artifact_uri=model_uri, dst_path=dst_path)

    model = Model.load(local_model_path)

    flavor_conf = model.flavors[FLAVOR_NAME]

    model_type = flavor_conf.get('model_type', 'tabular')

    autogluon_model_path = Path(local_model_path) / AUTODEPLOY_SUBPATH

    loader = get_model_loader(model_type)
    return loader(autogluon_model_path)


def _load_model_from_local_path(local_model_path: str | Path) -> Any:
    """Load AutoGluon model from local path without download_artifacts().

    This function loads a model that is already on local filesystem,
    without calling download_artifacts(). Used by PyFunc wrapper
    when the model is already downloaded.

    Args:
        local_model_path: Local path to model directory

    Returns:
        Loaded AutoGluon model instance

    Raises:
        ValueError: If model cannot be loaded or flavor configuration is invalid
    """
    model = Model.load(local_model_path)
    flavor_conf = model.flavors[FLAVOR_NAME]
    model_type = flavor_conf.get('model_type', 'tabular')
    autogluon_model_path = Path(local_model_path) / AUTODEPLOY_SUBPATH
    loader = get_model_loader(model_type)
    return loader(autogluon_model_path)


def get_model_loader(model_type: ModelTypeLiteral) -> Any:
    """Get the loader function for the given model type.

    Args:
        model_type: Type of AutoGluon model

    Returns:
        Loader function that takes a path and returns a loaded model

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == 'tabular':
        from autogluon.tabular import TabularPredictor  # noqa: WPS433

        return TabularPredictor.load
    if model_type == 'multimodal':
        from autogluon.multimodal import MultiModalPredictor  # noqa: WPS433

        return MultiModalPredictor.load
    if model_type == 'vision':
        from autogluon.vision import VisionPredictor  # noqa: WPS433

        return VisionPredictor.load
    if model_type == 'timeseries':
        from autogluon.timeseries import TimeSeriesPredictor  # noqa: WPS433

        return TimeSeriesPredictor.load

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        f"Supported: ['tabular', 'multimodal', 'vision', 'timeseries']",
    )
