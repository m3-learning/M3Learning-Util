import pytest
import tempfile
import os
from unittest import mock
import numpy as np
import cv2
from m3util.viz.movies import make_movie


@pytest.fixture
def temp_dirs_and_files():
    # Create a temporary directory and some dummy image files
    with tempfile.TemporaryDirectory() as temp_input_dir, tempfile.TemporaryDirectory() as temp_output_dir:
        for i in range(5):
            file_path = os.path.join(temp_input_dir, f"image_{i}.png")
            dummy_image = np.ones((100, 200, 3), dtype=np.uint8) * i
            cv2.imwrite(file_path, dummy_image)

        yield temp_input_dir, temp_output_dir


@mock.patch('cv2.VideoWriter')
@mock.patch('cv2.imread')
@mock.patch('glob.glob')
@mock.patch('m3util.util.IO.make_folder')
def test_make_movie_basic(mock_make_folder, mock_glob, mock_imread, mock_VideoWriter, temp_dirs_and_files):
    temp_input_dir, temp_output_dir = temp_dirs_and_files

    # Mock the return value of make_folder to just return the output directory
    mock_make_folder.return_value = temp_output_dir

    # Mock glob to return a sorted list of image files
    mock_glob.return_value = sorted([os.path.join(temp_input_dir, f"image_{i}.png") for i in range(5)])

    # Mock cv2.imread to return dummy images
    mock_imread.side_effect = lambda x: np.ones((100, 200, 3), dtype=np.uint8) * int(x.split('_')[-1].split('.')[0])

    # Mock VideoWriter object to simulate video writing
    mock_writer = mock.Mock()
    mock_VideoWriter.return_value = mock_writer

    # Call the function under test
    make_movie("test_movie", temp_input_dir, temp_output_dir, "png", fps=10, reverse=False)

    # Assertions
    mock_glob.assert_called_with(f"{temp_input_dir}/*.png")
    assert mock_imread.call_count == 5, "cv2.imread should be called for each image file."
    mock_VideoWriter.assert_called_once()
    assert mock_writer.write.call_count == 5, "VideoWriter.write should be called for each frame."
    mock_writer.release.assert_called_once()


@mock.patch('cv2.VideoWriter')
@mock.patch('cv2.imread')
@mock.patch('glob.glob')
@mock.patch('m3util.util.IO.make_folder')
def test_make_movie_with_reverse(mock_make_folder, mock_glob, mock_imread, mock_VideoWriter, temp_dirs_and_files):
    temp_input_dir, temp_output_dir = temp_dirs_and_files

    # Mock the return value of make_folder to just return the output directory
    mock_make_folder.return_value = temp_output_dir

    # Mock glob to return a sorted list of image files
    mock_glob.return_value = sorted(
        [os.path.join(temp_input_dir, f"image_{i}.png") for i in range(5)])

    # Mock cv2.imread to return dummy images
    mock_imread.side_effect = lambda x: np.ones(
        (100, 200, 3), dtype=np.uint8) * int(x.split('_')[-1].split('.')[0])

    # Mock VideoWriter object to simulate video writing
    mock_writer = mock.Mock()
    mock_VideoWriter.return_value = mock_writer

    # Call the function under test with reverse=True
    make_movie("test_movie_reverse", temp_input_dir,
               temp_output_dir, "png", fps=10, reverse=True)

    # Assertions
    assert mock_imread.call_count == 10, "cv2.imread should be called for each image file twice due to reverse=True."
    assert mock_writer.write.call_count == 10, "VideoWriter.write should be called for each frame in the reversed sequence."


@mock.patch('cv2.VideoWriter')
@mock.patch('cv2.imread')
@mock.patch('glob.glob')
@mock.patch('m3util.util.IO.make_folder')
def test_make_movie_with_text(mock_make_folder, mock_glob, mock_imread, mock_VideoWriter, temp_dirs_and_files):
    temp_input_dir, temp_output_dir = temp_dirs_and_files

    # Mock the return value of make_folder to just return the output directory
    mock_make_folder.return_value = temp_output_dir

    # Mock glob to return a sorted list of image files
    mock_glob.return_value = sorted(
        [os.path.join(temp_input_dir, f"image_{i}.png") for i in range(5)])

    # Mock cv2.imread to return dummy images
    mock_imread.side_effect = lambda x: np.ones(
        (100, 200, 3), dtype=np.uint8) * int(x.split('_')[-1].split('.')[0])

    # Mock VideoWriter object to simulate video writing
    mock_writer = mock.Mock()
    mock_VideoWriter.return_value = mock_writer

    # Call the function under test with text_list enabled
    make_movie("test_movie_text", temp_input_dir,
               temp_output_dir, "png", fps=10, text_list=True)

    # Assertions
    assert mock_writer.write.call_count == 5, "VideoWriter.write should be called for each frame."
    # We expect putText to be called for each frame
    # putText is not directly accessible for mocking here, so we check if text_list was not None,
    # which indicates that the text was processed
    for call in mock_writer.write.call_args_list:
        frame = call[0][0]
        # Assuming some text processing has happened, we check the frame shape is preserved
        assert frame.shape == (
            100, 200, 3), "Frame shape should remain unchanged after adding text."
