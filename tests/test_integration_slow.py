"""
Integration tests with real AutoGluon models.

These tests train actual models and test the full MLflow lifecycle.
Marked with @pytest.mark.slow for opt-out in CI/CD.
"""

import mlflow
import pytest

import mlflow_autogluon
from tests.utils import get_model_fixtures, get_model_predictions, get_pyfunc_input


@pytest.mark.slow
@pytest.mark.parametrize('model_type', ['tabular', 'multimodal', 'vision', 'timeseries'])
def test_full_lifecycle_train_log_load_predict(model_type, mlflow_tracking_uri, request):
    """Test full lifecycle of train, log, load, and predict for all model types."""
    model_fixture, data_fixture = get_model_fixtures(model_type, request)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    with mlflow.start_run():
        model_info = mlflow_autogluon.log_model(
            autogluon_model=model_fixture,
            artifact_path='model',
            model_type=model_type,
        )

    assert model_info.model_uri.startswith(('runs:/', 'models:/'))

    loaded_model = mlflow_autogluon.load_model(model_info.model_uri)

    predictions = get_model_predictions(loaded_model, model_type, data_fixture)
    assert predictions is not None
    assert len(predictions) > 0


@pytest.mark.slow
@pytest.mark.parametrize('model_type', ['tabular', 'multimodal', 'timeseries'])
def test_pyfunc_wrapper_with_real_model(model_type, mlflow_tracking_uri, request):
    """Test PyFunc wrapper with real trained model."""
    model_fixture, data_fixture = get_model_fixtures(model_type, request)

    mlflow.set_tracking_uri(mlflow_tracking_uri)

    with mlflow.start_run():
        model_info = mlflow_autogluon.log_model(
            autogluon_model=model_fixture,
            artifact_path='model',
            model_type=model_type,
        )

    pyfunc_model = mlflow.pyfunc.load_model(model_info.model_uri)

    input_data = get_pyfunc_input(model_type, data_fixture)
    predictions = pyfunc_model.predict(input_data)
    assert predictions is not None
    assert len(predictions) > 0


@pytest.mark.slow
@pytest.mark.parametrize('model_type', ['tabular', 'multimodal', 'vision', 'timeseries'])
def test_save_load_without_mlflow(model_type, tmp_path, request):
    """Test save_model and load_model without MLflow tracking."""
    model_fixture, _ = get_model_fixtures(model_type, request)

    save_path = tmp_path / 'saved_model'

    mlflow_autogluon.save_model(
        autogluon_model=model_fixture,
        path=str(save_path),
        model_type=model_type,
    )

    assert (save_path / 'MLmodel').exists()
    assert (save_path / 'model').exists()
    assert (save_path / 'autogluon_metadata.json').exists()

    loaded_model = mlflow_autogluon.load_model(str(save_path))
    assert loaded_model is not None


@pytest.mark.slow
def test_model_metadata_preserved(tmp_path, trained_tabular_model):
    """Test that model metadata is correctly saved and loaded."""
    import json

    save_path = tmp_path / 'saved_model'

    mlflow_autogluon.save_model(
        autogluon_model=trained_tabular_model,
        path=str(save_path),
        model_type='tabular',
    )

    metadata_path = save_path / 'autogluon_metadata.json'
    assert metadata_path.exists()

    with open(metadata_path) as f:
        metadata = json.load(f)

    assert metadata['model_type'] == 'tabular'
    assert 'model_class' in metadata
    assert 'model_module' in metadata
