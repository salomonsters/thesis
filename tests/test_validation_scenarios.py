import pytest
from pytest import approx
import numpy as np

from validation_scenarios import generate_start_positions, earth_r


def test_generate_start_positions():
    r = 1000
    x_start, y_start = generate_start_positions((0., 0.), r, np.radians([0, 10, 20, 30, 90]))
    assert np.allclose(np.sqrt((earth_r * (x_start - 0.)*np.cos(0.))**2 + (earth_r * (y_start - 0.))**2), r)
    assert x_start[0] == approx(0)
    assert y_start[0] == approx(r/earth_r)

    assert x_start[3] == approx(r/earth_r * 0.5)
    assert y_start[3] == approx(r/earth_r * np.sqrt(3/4))

    assert x_start[4] == approx(r / earth_r)
    assert y_start[4] == approx(0)

