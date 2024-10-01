import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset
from m3util.ml.inference import computeTime


class MockModel(torch.nn.Module):
    def forward(self, x):
        return x


@pytest.fixture
def mock_dataloader():
    data = torch.randn(100, 10)  # 100 samples, 10 features each
    dataset = TensorDataset(data)
    return DataLoader(dataset, batch_size=10)


@pytest.fixture
def mock_model():
    return MockModel()


def test_computeTime(mock_model, mock_dataloader):
    result = computeTime(
        mock_model, mock_dataloader, batch_size=10, device="cpu", write_to_file=True
    )
    assert isinstance(result, str), "Expected result to be a string"
    assert (
        "Avg execution time (ms):" in result
    ), "Expected result to contain 'Avg execution time (ms):'"


def test_computeTime_no_write(mock_model, mock_dataloader):
    result = computeTime(
        mock_model, mock_dataloader, batch_size=10, device="cpu", write_to_file=False
    )
    assert result is None, "Expected result to be None when write_to_file is False"
