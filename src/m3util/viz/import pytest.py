import pytest
from unittest import mock
from .images import display_image

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

@pytest.fixture
def mock_axis():
    fig, ax = plt.subplots()
    yield ax
    plt.close(fig)

@mock.patch("matplotlib.image.imread")
def test_display_image(mock_imread, mock_axis):
    # Mock the image read function to return a dummy image
    dummy_image = mock.Mock()
    mock_imread.return_value = dummy_image

    # Call the function under test
    display_image(mock_axis, "dummy_path.png")

    # Assertions
    mock_imread.assert_called_once_with("dummy_path.png")
    mock_axis.imshow.assert_called_once_with(dummy_image)
    mock_axis.axis.assert_called_once_with("off")