import pytest
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
from unittest.mock import MagicMock, patch
from m3util.ml.optimizers.TrustRegion import TRCG


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.linear = nn.Linear(10, 1, bias=False)

    def forward(self, x):
        return self.linear(x)


def simple_closure(model, data, target, part, total_parts, device):
    def closure_fn():
        start = part * len(data) // total_parts
        end = (part + 1) * len(data) // total_parts
        x = data[start:end].to(device)
        y = target[start:end].to(device)
        output = model(x)
        loss = nn.MSELoss()(output, y)
        return loss

    return closure_fn


@pytest.fixture
def sample_data():
    torch.manual_seed(0)
    data = torch.randn(100, 10)
    target = torch.randn(100, 1)
    return data, target


def test_trcg_initialization():
    model = SimpleModel()
    optimizer = TRCG(model, radius=0.1, device="cpu")
    assert isinstance(optimizer, Optimizer)
    assert optimizer.radius == 0.1
    assert optimizer.device == "cpu"


def test_trcg_step(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu")

    def closure(part=0, total_parts=1, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    initial_loss = closure().item()
    optimizer.step(closure)
    new_loss = closure().item()
    assert new_loss < initial_loss, "Loss did not decrease after optimization step"


def test_trcg_cgsolver(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu")

    def closure(part=0, total_parts=1, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    loss = closure()
    loss_grad = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    cnt_compute = 0
    direction, cg_iter, cg_term, cnt_compute = optimizer.CGSolver(
        loss_grad, cnt_compute, closure
    )
    assert cg_iter > 0, "CG iterations should be greater than zero"
    assert isinstance(direction, list), "Direction should be a list of tensors"


def test_trcg_compute_hessian_vector(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu")

    def closure(part=0, total_parts=1, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    loss = closure()
    direction = [param.data.clone() for param in model.parameters()]
    hessian_vector = optimizer.computeHessianVector(closure, direction)
    assert isinstance(hessian_vector, list), "Hessian-vector product should be a list"
    for hv in hessian_vector:
        assert (
            hv.shape == direction[0].shape
        ), "Shapes of Hessian-vector product do not match"



def test_trcg_multiple_steps(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu")
    losses = []

    def closure(part=0, total_parts=1, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    for _ in range(5):
        optimizer.step(closure)
        loss = closure().item()
        losses.append(loss)

    assert losses == sorted(losses, reverse=True), "Loss did not consistently decrease"


def test_trcg_with_closure_size(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu", closure_size=2)

    def closure(part=0, total_parts=2, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    initial_loss = sum(closure(part=i).item() for i in range(2))
    optimizer.step(closure)
    new_loss = sum(closure(part=i).item() for i in range(2))
    assert new_loss < initial_loss, "Loss did not decrease with closure_size > 1"


def test_trcg_findroot():
    model = SimpleModel()
    optimizer = TRCG(model, radius=1.0, device="cpu")
    x = [torch.tensor([0.5])]
    p = [torch.tensor([1.0])]
    alpha = optimizer.findroot(x, p)
    expected_alpha = (optimizer.radius - x[0].norm()).item() / p[0].norm().item()
    assert alpha == expected_alpha, "Alpha computed incorrectly in findroot"


def test_trcg_compute_loss(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu")

    def closure(part=0, total_parts=1, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    loss = optimizer.computeLoss(closure)
    expected_loss = closure().item()
    assert loss == expected_loss, "computeLoss did not return the correct loss value"


def test_trcg_compute_gradient(sample_data):
    model = SimpleModel()
    data, target = sample_data
    optimizer = TRCG(model, radius=0.1, device="cpu")

    def closure(part=0, total_parts=1, device="cpu"):
        return simple_closure(model, data, target, part, total_parts, device)()

    gradients = optimizer.computeGradient(closure)
    expected_gradients = torch.autograd.grad(closure(), model.parameters())
    for g1, g2 in zip(gradients, expected_gradients):
        assert torch.allclose(
            g1, g2
        ), "Computed gradients do not match expected gradients"


def test_trcg_compute_dot_product():
    model = SimpleModel()
    optimizer = TRCG(model, radius=1.0, device="cpu")
    v = [torch.tensor([1.0, 2.0])]
    z = [torch.tensor([3.0, 4.0])]
    dot_product = optimizer.computeDotProduct(v, z)
    expected_dot_product = torch.dot(v[0], z[0])
    assert (
        dot_product.item() == expected_dot_product.item()
    ), "Dot product computed incorrectly"


def test_trcg_compute_norm():
    model = SimpleModel()
    optimizer = TRCG(model, radius=1.0, device="cpu")
    v = [torch.tensor([3.0, 4.0])]
    norm = optimizer.computeNorm(v)
    expected_norm = torch.norm(v[0])
    assert norm.item() == expected_norm.item(), "Norm computed incorrectly"
