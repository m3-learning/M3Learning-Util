import pytest
from unittest.mock import patch
import matplotlib.pyplot as plt

# Assuming your code file is named `your_module.py`
from m3util.viz.images import display_image


@patch("m3util.viz.images.mpimg.imread")  # Mock the imread function from mpimg
def test_display_image(mock_imread):
    """
    Test display_image function to ensure the image is displayed properly
    and the axis is turned off.
    """
    # Mock image data to simulate an image read
    mock_imread.return_value = [[1, 2], [3, 4]]  # Dummy image data as a 2x2 matrix

    # Create a figure and axis for the test
    fig, ax = plt.subplots()

    # Call the display_image function with a mock image path
    display_image(ax, "dummy_image_path.png")

    # Ensure imread was called with the correct file path
    mock_imread.assert_called_once_with("dummy_image_path.png")

    # Check that imshow was called on the axis with the dummy image data
    assert ax.images[0].get_array().data.tolist() == [[1, 2], [3, 4]]

    plt.close(fig)  # Close the figure after testing
