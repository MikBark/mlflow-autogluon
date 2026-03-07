"""Model saving functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME

from mlflow_autogluon.constants import (
    AUTODEPLOY_METADATA_FILE,
    AUTODEPLOY_SUBPATH,
    FLAVOR_NAME,
    MODEL_PACKAGES,
)
from mlflow_autogluon.literals import ModelTypeLiteral
from mlflow_autogluon.requirements import get_default_conda_env


@dataclass(frozen=True)
class _MlflowConfig:
    """Configuration for MLflow model setup.

    Attributes:
        model_type: Type of AutoGluon model
        conda_env: Conda environment dict or path to yaml file
        extra_pip_requirements: Extra pip requirements to add
    """

    model_type: str
    conda_env: dict | str | None = None
    extra_pip_requirements: list[str] | None = None


def save_model(  # noqa: WPS201,WPS211,WPS213
    autogluon_model: Any | object,
    path: str,
    model_type: ModelTypeLiteral = 'tabular',
    mlflow_model: Model | None = None,
    conda_env: dict | str | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Save an AutoGluon model to a path on the local file system.

    Args:
        autogluon_model: AutoGluon model instance (e.g., TabularPredictor)
        path: Local path where model is to be saved
        model_type: Type of AutoGluon model ('tabular', 'multimodal', etc.)
        mlflow_model: MLflow model config to add to (creates new if None)
        conda_env: Conda environment dict or path to conda env yaml file
        pip_requirements: Override default pip requirements
        extra_pip_requirements: Extra pip requirements to add to defaults
        **kwargs: Additional arguments for AutoGluon-specific configuration

    Raises:
        MlflowException: If model_type is not supported or model lacks save() method
    """
    if model_type not in MODEL_PACKAGES:
        raise MlflowException(
            message=f"Unsupported model_type '{model_type}'.",
        )

    _validate_model(autogluon_model)
    path = _prepare_save_path(path)
    config = _MlflowConfig(model_type, conda_env, extra_pip_requirements)
    mlflow_model = _configure_mlflow_model(
        mlflow_model,
        path,
        config,
        kwargs,
    )

    _save_autogluon_model(autogluon_model, path, model_type)
    _write_metadata(autogluon_model, path, model_type)


def _validate_model(model: Any | object) -> None:
    """Validate that the model has required capabilities.

    Args:
        model: AutoGluon model instance

    Raises:
        MlflowException: If model lacks save() method
    """
    if not hasattr(model, 'save'):
        model_type_name = type(model).__name__
        raise MlflowException(
            message=(
                f"Model of type '{model_type_name}' must have a "
                "'save()' method. AutoGluon models typically have this method."
            ),
        )


def _prepare_save_path(path: str) -> Path:
    """Prepare and create the save directory.

    Args:
        path: Path where model should be saved

    Returns:
        Resolved Path object
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _configure_mlflow_model(
    mlflow_model: Model | None,
    path: Path,
    config: _MlflowConfig,
    kwargs: dict[str, Any],
) -> Model:
    """Configure MLflow model with flavors and save MLmodel file.

    Args:
        mlflow_model: Existing MLflow model or None
        path: Path where model is being saved
        config: MLflow configuration including model type and conda env
        kwargs: Additional arguments

    Returns:
        Configured MLflow Model instance
    """
    if mlflow_model is None:
        mlflow_model = Model()

    conda_env = config.conda_env
    if conda_env is None:
        conda_env = get_default_conda_env(
            model_type=config.model_type,
            additional_pip_requirements=config.extra_pip_requirements,
        )
    elif isinstance(conda_env, str):
        conda_env = json.loads(Path(conda_env).read_text())

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_type=config.model_type,
        autogluon_version=kwargs.get('autogluon_version'),
        predictor_metadata=kwargs.get('predictor_metadata', {}),
    )

    mlflow_model.add_flavor(
        'python_function',
        loader_module='mlflow_autogluon.pyfunc',
        model_type=config.model_type,
    )

    mlflow_model_file_path = path / MLMODEL_FILE_NAME
    mlflow_model.save(mlflow_model_file_path)

    return mlflow_model


def _save_autogluon_model(
    autogluon_model: Any | object,
    path: Path,
    model_type: str,
) -> None:
    """Save the AutoGluon model to the specified path.

    Args:
        autogluon_model: AutoGluon model instance
        path: Path where model should be saved
        model_type: Type of AutoGluon model
    """
    autogluon_model_path = path / AUTODEPLOY_SUBPATH

    if model_type == 'tabular':
        _save_tabular_model(autogluon_model, autogluon_model_path)
    else:
        autogluon_model.save(str(autogluon_model_path))


def _save_tabular_model(
    autogluon_model: Any | object,
    autogluon_model_path: Path,
) -> None:
    """Save tabular model.

    Args:
        autogluon_model: TabularPredictor instance
        autogluon_model_path: Path where model should be saved
    """
    if autogluon_model_path.exists():
        shutil.rmtree(autogluon_model_path)
    autogluon_model.save(str(autogluon_model_path))


def _write_metadata(
    autogluon_model: Any | object,
    path: Path,
    model_type: str,
) -> None:
    """Write model metadata to JSON file.

    Args:
        autogluon_model: AutoGluon model instance
        path: Path where model is saved
        model_type: Type of AutoGluon model
    """
    metadata = {
        'model_type': model_type,
        'model_class': type(autogluon_model).__name__,
        'model_module': type(autogluon_model).__module__,
    }

    if model_type == 'tabular' and hasattr(autogluon_model, 'predict'):
        metadata['supports_predict_proba'] = hasattr(
            autogluon_model,
            'predict_proba',
        )

    metadata_file_path = path / AUTODEPLOY_METADATA_FILE
    with metadata_file_path.open('w') as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
