"""Test utilities for MLflow-AutoGluon tests."""


def get_model_fixtures(model_type: str, request):
    """Get trained model and corresponding data fixture for model type."""
    return (
        request.getfixturevalue(f'trained_{model_type}_model'),
        request.getfixturevalue(f'sample_{model_type}_data'),
    )


def get_model_predictions(model, model_type: str, data_fixture):
    """Get predictions from model using appropriate input format."""
    if model_type in ['tabular', 'multimodal', 'timeseries']:
        train_df, test_df, label = data_fixture
        if model_type == 'tabular':
            return model.predict(test_df.drop(columns=[label]))
        return model.predict(test_df)
    else:
        _, test_dir = data_fixture
        return model.predict(str(test_dir))


# TODO: Replace the pyfunc input with a fixture everywhere.
# Use pytest-lazy-fixtures==1.1.2
def get_pyfunc_input(model_type: str, data_fixture):
    """Get input data for PyFunc prediction based on model type."""
    if model_type == 'tabular':
        train_df, test_df, label = data_fixture
        return test_df.drop(columns=[label])
    elif model_type == 'multimodal':
        train_df, test_df, _ = data_fixture
        return test_df
    elif model_type == 'vision':
        _, test_dir = data_fixture
        return test_dir
    else:
        train_df, test_df, _ = data_fixture
        return test_df
