"""Limit proposal from measured resting peaks."""

import numpy as np

from force_baseline import propose_limits


def test_limit_is_margin_times_peak():
    peak = np.array([0.01, 0.05, 0.02])
    limits = propose_limits(peak, k=4.0)
    assert np.allclose(limits, [0.04, 0.2, 0.08])


def test_quiet_channel_gets_floored():
    peak = np.array([0.05, 1e-6])
    limits = propose_limits(peak, k=4.0, floor_ratio=0.15)
    assert limits[0] == 0.2
    assert limits[1] == 0.15 * 0.05  # floor keeps the channel positive
    assert np.all(limits > 0)


def test_uniform_channels_scale_together():
    peak = np.full(22, 0.03)
    limits = propose_limits(peak, k=4.0)
    assert np.allclose(limits, 0.12)
