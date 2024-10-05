import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch
from m3util.ml.inference import computeTime


class MockModel(nn.Module):
    def __init__(self):
        super(MockModel, self).__init__()
        self.linear = nn.Linear(10, 5)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_model():
    return MockModel()


@pytest.fixture
def mock_dataloader():
    # Create some mock data
    data = torch.randn(50, 10)  # 50 samples, 10 features
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)
    return dataloader


def test_computeTime_write_to_file(mock_model, mock_dataloader):
    # Mock time.time() to have consistent timing
    with patch("time.time", side_effect=[0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        with patch("builtins.print") as mock_print:
            result = computeTime(
                mock_model,
                mock_dataloader,
                batch_size=10,
                device="cpu",
                write_to_file=True,
            )
            # Check that the result is a string containing the average time
            assert isinstance(result, str)
            assert "Avg execution time (ms):" in result

            # Check that print statements were called
            mock_print.assert_any_call(
                "Mean execution time computed for 5 batches of size 10"
            )
            # The total execution time should be the sum of time_spent
            total_time_spent = 0.1 * 5  # Since time.time() increments by 0.1
            mock_print.assert_any_call(
                f"Total execution time (s): {total_time_spent:.2f} "
            )


def test_computeTime_no_write(mock_model, mock_dataloader):
    # Mock time.time() to have consistent timing
    with patch("time.time", side_effect=[0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        with patch("builtins.print") as mock_print:
            result = computeTime(
                mock_model,
                mock_dataloader,
                batch_size=10,
                device="cpu",
                write_to_file=False,
            )
            # Since write_to_file is False, result should be None
            assert result is None

            # Check that print statements were called
            mock_print.assert_any_call(
                "Mean execution time computed for 5 batches of size 10"
            )
            total_time_spent = 0.1 * 5  # Since time.time() increments by 0.1
            mock_print.assert_any_call(
                f"Total execution time (s): {total_time_spent:.2f} "
            )


def test_computeTime_data_format(mock_model):
    # Create a dataloader where data is a tuple (inputs, labels)
    data = torch.randn(50, 10)
    labels = torch.randint(0, 5, (50,))
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    with patch("time.time", side_effect=[0, 0.1, 0.2, 0.3, 0.4, 0.5]):
        with patch("builtins.print"):
            # Modify computeTime to handle data as tuple
            result = computeTime(
                mock_model, dataloader, batch_size=10, device="cpu", write_to_file=False
            )
            # Ensure no exceptions occur


def test_computeTime_cuda(mock_model, mock_dataloader):
    # Only run this test if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        mock_model.to(device)
        with patch("time.time", side_effect=[0, 0.1, 0.2, 0.3, 0.4, 0.5]):
            with patch("torch.cuda.synchronize") as mock_sync:
                with patch("builtins.print"):
                    result = computeTime(
                        mock_model,
                        mock_dataloader,
                        batch_size=10,
                        device=device,
                        write_to_file=False,
                    )
                    # Check that torch.cuda.synchronize was called
                    assert mock_sync.call_count == 5  # Should be called once per batch
    else:
        pytest.skip("CUDA is not available")


def test_computeTime_exception_handling(mock_model, mock_dataloader):
    # Test if the function handles exceptions during model inference
    def faulty_forward(x):
        raise RuntimeError("Intentional error for testing")

    mock_model.forward = faulty_forward

    with patch("builtins.print") as mock_print:
        with pytest.raises(RuntimeError):
            computeTime(
                mock_model,
                mock_dataloader,
                batch_size=10,
                device="cpu",
                write_to_file=False,
            )


def test_computeTime_output_values(mock_model, mock_dataloader):
    # Test if the printed average times are correct given mocked time.time()
    time_values = [0, 0.01, 0.03, 0.06, 0.1, 0.15]
    with patch("time.time", side_effect=time_values):
        with patch("builtins.print") as mock_print:
            computeTime(
                mock_model,
                mock_dataloader,
                batch_size=10,
                device="cpu",
                write_to_file=False,
            )
            # Calculate expected times
            time_spent = [
                time_values[i + 1] - time_values[i]
                for i in range(1, len(time_values) - 1)
            ]
            averaged_time = np.mean(time_spent) * 1000
            std_time = np.std(time_spent) * 1000
            # Check that the print statements contain the correct averaged_time and std_time
            expected_avg_time_str = f"Average execution time per batch (ms): {averaged_time:.6f} ± {std_time:.6f}"
            mock_print.assert_any_call(expected_avg_time_str)


def test_computeTime_zero_batches(mock_model):
    # Create an empty dataloader
    data = torch.randn(0, 10)
    dataset = TensorDataset(data)
    dataloader = DataLoader(dataset, batch_size=10)

    with patch("builtins.print") as mock_print:
        computeTime(
            mock_model, dataloader, batch_size=10, device="cpu", write_to_file=False
        )
        # Check that the function handles zero batches gracefully
        mock_print.assert_any_call(
            "Mean execution time computed for 0 batches of size 10"
        )
