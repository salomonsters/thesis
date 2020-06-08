import pytest
from pytest import approx
import numpy as np

from validation_scenarios import generate_start_positions, earth_r, generate_tracks_within_box


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


def test_generate_tracks_within_box():
    noise = np.array([[0., 0.], [-1, -1.], [1, -1], [0, 1]])
    width, height = np.array((3., 20.))*1852
    cog = np.array((0., 0.))
    x_start, y_start, heading = generate_tracks_within_box(cog, width, height, 0, 4, noise=noise)
    assert x_start[0] == cog[0]
    assert x_start[1] == cog[0] - 0.5*width
    assert x_start[2] == cog[0] + 0.5*width
    assert x_start[3] == cog[0]

    assert np.allclose(y_start[:], cog[1] - 0.5*height)

    assert heading[0] == 0
    assert heading[1] == 0
    assert heading[2] == 360 - np.rad2deg(np.arctan(3./20.))
    assert heading[3] == np.rad2deg(np.arctan(1.5/20.))

    x_start, y_start, heading = generate_tracks_within_box(cog, width, height, 180, 4, noise=noise)
    assert np.allclose(x_start, cog[0] + np.array([0, 0.5*width, -0.5*width, 0]))
    assert np.allclose(y_start[:], cog[1] + 0.5*height)
    assert np.allclose(heading, (180 + np.array([0, 0, - np.rad2deg(np.arctan(3./20.)), np.rad2deg(np.arctan(1.5/20.))])) % 360)

    x_start, y_start, heading = generate_tracks_within_box(cog, width, height, -90, 4, noise=noise)
    assert np.allclose(y_start, cog[0] - np.array([0, 0.5*width, -0.5*width, 0]))
    assert np.allclose(x_start[:], cog[1] + 0.5*height)
    assert np.allclose(heading, (-90 + np.array([0, 0, - np.rad2deg(np.arctan(3./20.)), np.rad2deg(np.arctan(1.5/20.))])) % 360)

