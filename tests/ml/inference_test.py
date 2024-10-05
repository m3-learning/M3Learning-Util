import pytest
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from unittest.mock import patch
from m3util.ml.inference import computeTime
import itertools


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
    # Create enough time values for all calls to time.time()
    time_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with patch("time.time", side_effect=time_values):
        with patch("builtins.print") as mock_print:
            result = computeTime(
                mock_model,
                mock_dataloader,
                batch_size=10,
                device="cpu",
                write_to_file=True,
            )
            assert isinstance(result, str)
            assert "Avg execution time (ms):" in result

            # Verify that the print statements were called
            mock_print.assert_any_call(
                "Mean execution time computed for 5 batches of size 10"
            )


def test_computeTime_data_format(mock_model):
    data = torch.randn(50, 10)
    labels = torch.randint(0, 5, (50,))
    dataset = TensorDataset(data, labels)
    dataloader = DataLoader(dataset, batch_size=10)

    time_values = itertools.count(start=0, step=0.1)
    with patch("time.time", side_effect=time_values):
        with patch("builtins.print"):
            result = computeTime(
                mock_model, dataloader, batch_size=10, device="cpu", write_to_file=False
            )
            # Ensure no exceptions occur


def test_computeTime_output_values(mock_model, mock_dataloader):
    # Provide enough time values to prevent StopIteration
    time_values = [0, 0.01, 0.03, 0.06, 0.1, 0.15, 0.21, 0.28, 0.36, 0.45]
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
                for i in range(0, len(time_values) - 1, 2)
            ]
            averaged_time = np.mean(time_spent) * 1000
            std_time = np.std(time_spent) * 1000
            # Check that the print statements contain the correct averaged_time and std_time
            expected_avg_time_str = f"Average execution time per batch (ms): {averaged_time:.6f} ± {std_time:.6f}"
            mock_print.assert_any_call(expected_avg_time_str)
