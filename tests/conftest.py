"""Pytest fixtures for MLflow-AutoGluon tests."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_path():
    """Provide temporary path for tests."""
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp)


@pytest.fixture
def mlflow_tracking_uri(tmp_path):
    """Provide temporary MLflow tracking URI."""
    return f'file://{tmp_path}/mlruns'


@pytest.fixture
def sample_data():
    """Provide sample data for prediction tests."""
    return pd.DataFrame(
        {
            'feature_0': [1.0, 2.0, 3.0],
            'feature_1': [0.5, 1.5, 2.5],
            'feature_2': [0.1, 0.2, 0.3],
        }
    )


@pytest.fixture
def mock_test_data():
    """Provide mock test data list for unit tests."""
    return [1, 2, 3]


@pytest.fixture
def sample_tabular_data():
    """Generate small synthetic tabular dataset."""
    try:
        from sklearn.datasets import make_classification
    except ImportError:
        pytest.skip('scikit-learn not installed')

    X, y = make_classification(n_samples=100, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
    df['target'] = y
    train_df = df[:80]
    test_df = df[80:]
    return train_df, test_df, 'target'


@pytest.fixture
def sample_multimodal_data():
    """Generate small synthetic multimodal dataset."""
    train_df = pd.DataFrame(
        {
            'text': ['positive'] * 40 + ['negative'] * 40,
            'numerical': np.random.randn(80),
            'label': [1] * 40 + [0] * 40,
        }
    )
    test_df = pd.DataFrame(
        {
            'text': ['positive', 'negative', 'positive', 'negative'],
            'numerical': [0.5, -0.5, 0.3, -0.3],
        }
    )
    return train_df, test_df, 'label'


@pytest.fixture
def sample_vision_data(tmp_path):
    """Generate small synthetic vision dataset."""
    try:
        from PIL import Image
    except ImportError:
        pytest.skip('pillow not installed')

    train_dir = tmp_path / 'vision_train'
    test_dir = tmp_path / 'vision_test'

    for label in ['cat', 'dog']:
        (train_dir / label).mkdir(parents=True, exist_ok=True)
        (test_dir / label).mkdir(parents=True, exist_ok=True)

        for i in range(5):
            img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
            img = Image.fromarray(img_array)
            img.save(train_dir / label / f'{i}.png')

        img_array = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(test_dir / label / 'test.png')

    return train_dir, test_dir


@pytest.fixture
def sample_timeseries_data():
    """Generate small synthetic timeseries dataset."""
    train_df = pd.DataFrame(
        {
            'item_id': ['A'] * 48 + ['B'] * 48,
            'timestamp': pd.date_range('2020-01-01', periods=48, freq='H').tolist() * 2,
            'target': range(96),
        }
    )
    test_df = pd.DataFrame(
        {
            'item_id': ['A', 'B'],
            'timestamp': [pd.Timestamp('2020-01-03'), pd.Timestamp('2020-01-03')],
            'target': [np.nan, np.nan],
        }
    )
    return train_df, test_df, 'target'


@pytest.fixture
def trained_tabular_model(sample_tabular_data, tmp_path):
    """Train minimal TabularPredictor for testing."""
    pytest.importorskip('autogluon.tabular')

    from autogluon.tabular import TabularPredictor

    train_df, _, label = sample_tabular_data
    model_path = tmp_path / 'tabular_model'

    predictor = TabularPredictor(
        label=label,
        path=str(model_path),
    ).fit(
        train_df,
        presets='medium_quality',
        time_limit=30,
        hyperparameters={'RF': {'n_estimators': 2}},
    )
    return predictor


@pytest.fixture
def trained_multimodal_model(sample_multimodal_data, tmp_path):
    """Train minimal MultiModalPredictor for testing."""
    pytest.importorskip('autogluon.multimodal')

    from autogluon.multimodal import MultiModalPredictor

    train_df, _, label = sample_multimodal_data
    model_path = tmp_path / 'multimodal_model'

    predictor = MultiModalPredictor(
        label=label,
        path=str(model_path),
    ).fit(
        train_df,
        time_limit=60,
    )
    return predictor


@pytest.fixture
def trained_vision_model(sample_vision_data, tmp_path):
    """Train minimal VisionPredictor for testing."""
    pytest.importorskip('autogluon.vision')

    from autogluon.vision import ImagePredictor

    train_dir, _ = sample_vision_data
    model_path = tmp_path / 'vision_model'

    predictor = ImagePredictor(
        path=str(model_path),
    ).fit(
        str(train_dir),
        problem_type='classification',
        time_limit=60,
        hyperparameters={'epochs': 1},
    )
    return predictor


@pytest.fixture
def trained_timeseries_model(sample_timeseries_data, tmp_path):
    """Train minimal TimeSeriesPredictor for testing."""
    pytest.importorskip('autogluon.timeseries')

    from autogluon.timeseries import TimeSeriesPredictor

    train_df, _, label = sample_timeseries_data
    model_path = tmp_path / 'timeseries_model'

    predictor = TimeSeriesPredictor(
        target=label,
        path=str(model_path),
        prediction_length=2,
    ).fit(
        train_df,
        time_limit=30,
    )
    return predictor
