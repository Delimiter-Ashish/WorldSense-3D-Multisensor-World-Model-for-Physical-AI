import numpy as np

from worldsense3d.sim import SimulationConfig
from worldsense3d.sim.counterfactual import CounterfactualEpisodeGenerator


def test_counterfactual_pair_shapes():
    cfg = SimulationConfig(
        min_objects=3,
        max_objects=3,
        frames=6,
        context_frames=3,
        frame_stride=2,
        image_size=32,
        force_duration_steps=3,
    )

    gen = CounterfactualEpisodeGenerator(cfg, seed=7)

    try:
        pair = gen.generate_pair()
    finally:
        gen.close()

    assert pair["factual_rgb"].shape == (6, 32, 32, 3)
    assert pair["counterfactual_rgb"].shape == (6, 32, 32, 3)

    assert pair["factual_states"].shape == (6, 3, 13)
    assert pair["counterfactual_states"].shape == (6, 3, 13)

    assert pair["intervention_code"].item() in {0, 1, 2, 3}
    assert np.isfinite(pair["mean_target_shift"]).all()
