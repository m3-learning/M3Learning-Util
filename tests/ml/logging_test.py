import pytest
from unittest.mock import patch, mock_open, patch
from m3util.util.IO import append_to_csv
from m3util.ml.logging import write_csv

@patch("builtins.open", new_callable=mock_open)
@patch("os.path.isfile")
def test_write_csv_file_creation(mock_isfile, mock_open_file):
    # Mocking os.path.isfile to simulate the file does not exist
    mock_isfile.return_value = False

    # Test data
    write_CSV = "test_output.csv"
    path = "/some/path"
    model_name = "test_model"
    optimizer_name = "adam"
    i = 1
    noise = 0.1
    epochs = 50
    total_time = 120.5
    train_loss = 0.05
    batch_size = 32
    loss_func = "MSELoss"
    seed = 42
    stoppage_early = True
    model_updates = 100
    filename = "test_file.pth"

    # Call the write_csv function
    write_csv(
        write_CSV,
        path,
        model_name,
        optimizer_name,
        i,
        noise,
        epochs,
        total_time,
        train_loss,
        batch_size,
        loss_func,
        seed,
        stoppage_early,
        model_updates,
        filename
    )

    # Check if open was called with the correct file path and mode
    mock_open_file.assert_called_once_with(f"{path}/{write_CSV}", "a", newline="")

    # Check that write operations happened as expected
    handle = mock_open_file()
    assert handle.write.called

def test_write_csv_no_file():
    # Call the function with write_CSV=None to ensure it does nothing
    result = write_csv(
        None,
        "/some/path",
        "test_model",
        "adam",
        1,
        0.1,
        50,
        120.5,
        0.05,
        32,
        "MSELoss",
        42,
        True,
        100,
        "test_file.pth"
    )
    # Since there's no return and no file should be written, we check that result is None
    assert result is None
