"""PyFunc wrapper for AutoGluon models.

This module provides a PythonModel-based wrapper that enables AutoGluon models
to be used with mlflow.pyfunc.load_model() for standardized inference.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
from mlflow.pyfunc import PythonModel

from mlflow_autogluon.literals import PredictMethodLiteral
from mlflow_autogluon.load import _load_model_from_local_path, load_model
from mlflow_autogluon.pyfunc.input_parser import parse_input
from mlflow_autogluon.pyfunc.output_formatter import format_output


class AutoGluonModelWrapper(PythonModel):
    """PyFunc wrapper for AutoGluon models.

    This wrapper enables AutoGluon models to be loaded via mlflow.pyfunc.load_model()
    and provides a standardized predict() interface that accepts pandas DataFrames.
    """

    def __init__(
        self,
        path: str | Path | None = None,
        autogluon_model: Any = None,
    ) -> None:
        """Initialize the wrapper.

        Args:
            path: Path to saved model directory
            autogluon_model: Pre-loaded AutoGluon model instance
        """
        if autogluon_model is not None:
            self._model = autogluon_model
        elif path is not None:
            self.model_path = path
            self._model = None
        else:
            raise ValueError('Either path or autogluon_model must be provided')

    def load_context(self, context: Any) -> None:
        """Load the AutoGluon model from the artifact path.

        This method is called by MLflow before predict().

        Args:
            context: MLflow context containing artifact path
        """
        if self._model is None:
            model_path = getattr(context, 'artifacts', self._model_path)
            path_obj = Path(model_path)

            # Check if local path with MLmodel file - load directly
            if (path_obj / 'MLmodel').exists():
                self._model = _load_model_from_local_path(model_path)
            else:
                self._model = load_model(model_path)

    def predict(
        self,
        context: Any,
        model_input: pd.DataFrame | dict[str, Any],
        # TODO: replace `params` default with empty immutable dict (MappingProxyType)
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame | dict[str, Any] | list[Any]:
        """
        Generate predictions using the AutoGluon model.

        Args:
            context: MLflow context
            model_input: Input data as pandas DataFrame or dict
            params: Optional prediction parameters:
                - 'predict_method': 'predict' (default) or 'predict_proba'
                - 'as_multiclass': For binary classification, return both class probs
                - 'as_pandas': Return pandas DataFrame (default) or dict/list

        Returns:
            Predictions as DataFrame, dict, or list depending on params

        Raises:
            ValueError: If predict_method is invalid or not supported
        """
        if self._model is None:
            self.load_context(context)
        params = params or {}
        parsed_input = parse_input(model_input)
        output = self._execute_prediction(
            parsed_input, params.get('predict_method', 'predict'), params
        )
        return format_output(output, params.get('as_pandas', True))

    def _execute_prediction(
        self,
        model_input: pd.DataFrame,
        method: PredictMethodLiteral,
        params: dict[str, Any],
    ) -> Any:
        """Execute prediction using the specified method.

        Args:
            model_input: Parsed input DataFrame
            method: Prediction method to use
            params: Additional prediction parameters

        Returns:
            Prediction output

        Raises:
            ValueError: If method is invalid or not supported by the model
        """
        if method not in {'predict', 'predict_proba', 'predict_multi'}:
            raise ValueError(
                f"Invalid predict_method '{method}'. "
                'Must be one of: predict, predict_proba, predict_multi',
            )
        if method == 'predict_proba' and not hasattr(self._model, 'predict_proba'):
            model_name = type(self._model).__name__
            raise ValueError(f'Model {model_name} does not support predict_proba')
        if method == 'predict_multi' and not hasattr(self._model, 'predict_multi'):
            model_name = type(self._model).__name__
            raise ValueError(f'Model {model_name} does not support predict_multi')

        if method == 'predict':
            return self._model.predict(model_input)
        if method == 'predict_proba':
            as_multiclass = params.get('as_multiclass', False)
            return self._model.predict_proba(model_input, as_multiclass=as_multiclass)
        return self._model.predict_multi(model_input)


class _PyFuncWrapper:
    """Simple wrapper for MLflow PyFunc loading.

    This class provides the predict(data, params=None) signature expected
    by MLflow's internal pyfunc loader, delegating to AutoGluonModelWrapper.
    """

    def __init__(self, path: str) -> None:
        """Initialize the wrapper.

        Args:
            path: Path to saved model directory
        """
        self._wrapper = AutoGluonModelWrapper(path)

    def predict(
        self,
        data: pd.DataFrame | dict[str, Any],
        params: dict[str, Any] | None = None,
    ) -> pd.DataFrame | dict[str, Any] | list[Any]:
        """
        Generate predictions using the AutoGluon model.

        Args:
            data: Input data as pandas DataFrame or dict
            params: Optional prediction parameters

        Returns:
            Predictions as DataFrame, dict, or list depending on params
        """
        context = SimpleNamespace(artifacts=self._wrapper.model_path)
        return self._wrapper.predict(context, data, params)


def _load_pyfunc(path: str) -> Any:
    """
    Load AutoGluon model as PyFunc.

    This is used internally by MLflow when loading model with pyfunc flavor.

    Args:
        path: Local path to model directory

    Returns:
        PyFunc-compatible wrapper instance
    """
    return _PyFuncWrapper(path)
