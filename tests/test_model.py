import torch

from worldsense3d.models import MultiSensorWorldModel


def test_model_output_shape_and_finiteness():
    model = MultiSensorWorldModel(
        hidden_dim=48,
        visual_dim=32,
        temporal_layers=1,
        scene_layers=1,
        num_heads=4,
        max_objects=5,
        future_frames=5,
    )
    b, t, n, h, w = 2, 4, 5, 64, 64
    rgb = torch.randn(b, t, 3, h, w)
    depth = torch.randn(b, t, 1, h, w)
    state = torch.randn(b, n, 13)
    action = torch.randn(b, 6)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.float32)
    out = model(rgb, depth, state, action, mask)
    assert out.shape == (b, n, 5, 3)
    assert torch.isfinite(out).all()
