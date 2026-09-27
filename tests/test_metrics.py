import torch

from worldsense3d.metrics import trajectory_metrics


def test_zero_error_metrics():
    target = torch.zeros(2, 3, 4, 3)
    pred = target.clone()
    mask = torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.float32)
    target_index = torch.tensor([0, 2])
    m = trajectory_metrics(pred, target, mask, target_index)
    assert all(torch.isclose(v, torch.tensor(0.0)) for v in m.values())
