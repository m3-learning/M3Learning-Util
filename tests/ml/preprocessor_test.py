import pytest
import numpy as np
from m3util.ml.preprocessor import GlobalScaler  

@pytest.fixture
def sample_data():
    """Fixture to create a sample numpy array for testing."""
    return np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

@pytest.fixture
def scaler():
    """Fixture to create a GlobalScaler instance."""
    return GlobalScaler()

def test_fit(sample_data, scaler):
    """Test the fit method of GlobalScaler."""
    scaler.fit(sample_data)
    assert np.isclose(scaler.mean, 5.0), "Mean value is incorrect"
    assert np.isclose(scaler.std, np.std(sample_data)), "Standard deviation is incorrect"

def test_transform(sample_data, scaler):
    """Test the transform method of GlobalScaler."""
    scaler.fit(sample_data)
    transformed_data = scaler.transform(sample_data)
    expected_data = (sample_data - scaler.mean) / scaler.std
    np.testing.assert_array_almost_equal(transformed_data, expected_data, decimal=6)

def test_fit_transform(sample_data, scaler):
    """Test the fit_transform method of GlobalScaler."""
    transformed_data = scaler.fit_transform(sample_data)
    expected_data = (sample_data - np.mean(sample_data)) / np.std(sample_data)
    np.testing.assert_array_almost_equal(transformed_data, expected_data, decimal=6)

def test_inverse_transform(sample_data, scaler):
    """Test the inverse_transform method of GlobalScaler."""
    transformed_data = scaler.fit_transform(sample_data)
    inverse_data = scaler.inverse_transform(transformed_data)
    np.testing.assert_array_almost_equal(inverse_data, sample_data, decimal=6)

def test_transform_without_fit(sample_data, scaler):
    """Test the transform method without fitting to ensure it raises an error."""
    with pytest.raises(AttributeError):
        scaler.transform(sample_data)

def test_inverse_transform_without_fit(sample_data, scaler):
    """Test the inverse_transform method without fitting to ensure it raises an error."""
    with pytest.raises(AttributeError):
        scaler.inverse_transform(sample_data)
