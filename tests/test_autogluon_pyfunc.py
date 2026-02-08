"""PyFunc wrapper tests for AutoGluon integration."""

import tempfile
from unittest.mock import MagicMock, Mock

import pandas as pd
import pytest

from mlflow_autogluon.pyfunc import AutoGluonModelWrapper


def test_pyfunc_wrapper_init_with_model():
    """Test PyFunc wrapper initialization with pre-loaded model."""
    mock_model = Mock()

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    assert wrapper.predict(None, pd.DataFrame()) is not None


def test_pyfunc_wrapper_init_with_path():
    """Test PyFunc wrapper initialization with path."""
    with tempfile.TemporaryDirectory() as tmp:
        wrapper = AutoGluonModelWrapper(path=tmp)
        assert wrapper._model is None


def test_pyfunc_wrapper_init_needs_argument():
    """Test PyFunc wrapper requires either path or model."""
    with pytest.raises(ValueError) as exc_info:
        AutoGluonModelWrapper()
    assert 'Either path or autogluon_model must be provided' in str(exc_info.value)


def test_pyfunc_predict_with_params():
    """Test PyFunc predict() with params."""
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([1, 2, 3])

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    input_data = pd.DataFrame({'a': [1, 2, 3]})
    result = wrapper.predict(None, input_data, params={'predict_method': 'predict'})

    mock_model.predict.assert_called_once()
    assert isinstance(result, pd.Series)


def test_pyfunc_predict_invalid_method():
    """Test PyFunc predict() with invalid predict_method."""
    mock_model = MagicMock()

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    input_data = pd.DataFrame({'a': [1, 2, 3]})

    with pytest.raises(ValueError) as exc_info:
        wrapper.predict(None, input_data, params={'predict_method': 'invalid'})

    assert 'Invalid predict_method' in str(exc_info.value)


def test_pyfunc_predict_predict_proba_not_supported():
    """Test PyFunc predict() when model doesn't support predict_proba."""
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([1, 2, 3])
    del mock_model.predict_proba

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    input_data = pd.DataFrame({'a': [1, 2, 3]})

    with pytest.raises(ValueError) as exc_info:
        wrapper.predict(None, input_data, params={'predict_method': 'predict_proba'})

    assert 'does not support predict_proba' in str(exc_info.value)


def test_pyfunc_predict_with_dict_input():
    """Test PyFunc predict() with dict input."""
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([1, 2, 3])

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    input_data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
    result = wrapper.predict(None, input_data)

    mock_model.predict.assert_called_once()
    assert isinstance(result, pd.Series)


def test_pyfunc_predict_with_dataframe_split_format():
    """Test PyFunc predict() with MLflow REST API format."""
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.Series([1, 2, 3])

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    input_data = {
        'dataframe_split': {
            'data': [[1, 4], [2, 5], [3, 6]],
            'columns': ['a', 'b'],
        }
    }
    result = wrapper.predict(None, input_data)

    mock_model.predict.assert_called_once()
    assert isinstance(result, pd.Series)


def test_pyfunc_predict_as_pandas_false():
    """Test PyFunc predict() with as_pandas=False."""
    mock_model = MagicMock()
    mock_model.predict.return_value = pd.DataFrame({'pred': [1, 2, 3]})

    wrapper = AutoGluonModelWrapper(autogluon_model=mock_model)

    input_data = pd.DataFrame({'a': [1, 2, 3]})
    result = wrapper.predict(None, input_data, params={'as_pandas': False})

    assert isinstance(result, list)
