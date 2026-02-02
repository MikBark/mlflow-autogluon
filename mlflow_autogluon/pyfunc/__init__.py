"""PyFunc wrapper module for AutoGluon models."""

from mlflow_autogluon.pyfunc.pyfunc import AutoGluonModelWrapper, _load_pyfunc

__all__ = ['AutoGluonModelWrapper', '_load_pyfunc']
