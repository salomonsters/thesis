import math
import pytest

import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd

from timesnapshots import pairwise_range, range_rate, pairwise_range_rate, cut_interval

@pytest.fixture
def range_matrix():
    return np.array([[0, 0],
                     [0, 3],
                     [4, 0]],
                    dtype='float')

@pytest.fixture
def range_rate_matrix():
    return np.array([[0, 0, 100, 90],
                     [3000, 0, 100, 270],
                     [6000, 0, 200, 90],
                     [3000, 4000, 100, 270]],
                    dtype='float')


def test_range(range_matrix):
    Y = pairwise_range(range_matrix)
    assert np.allclose(Y, [3, 4, 5])


def test_range_rate(range_rate_matrix):
    X = range_rate_matrix
    assert range_rate(X[0, :], X[1, :]) == pytest.approx(200)
    assert range_rate(X[0, :], X[2, :]) == pytest.approx(-100)
    assert range_rate(X[0, :], X[3, :]) == pytest.approx(120)
    assert range_rate(X[1, :], X[2, :]) == pytest.approx(-300)
    assert range_rate(X[1, :], X[3, :]) == pytest.approx(0)
    assert range_rate(X[2, :], X[3, :]) == pytest.approx(-180)


def test_range_rate_pairwise(range_rate_matrix):
    Y = pairwise_range_rate(range_rate_matrix)
    assert np.allclose(Y, [200, -100, 120, -300, 0, -180])


def test_group_per_interval():
    df = pd.DataFrame.from_records([[0, 0], [0, 1], [1, 2], [2, 3]], columns=['ts', 'x'])
    dt = 1  #seconds
    assert len(cut_interval(df, dt).index.categories) == 3
    df = pd.DataFrame.from_records([[0, 0], [5, 1], [10, 2], [10.1, 2.1], [40, 18]], columns=['ts', 'x'])
    dt = 5  #seconds
    df_cut = cut_interval(df, dt)
    assert df_cut.index[-1].right == 45
    value_counts = df_cut.index.value_counts(sort=False)
    assert value_counts.iloc[0] == 1
    assert value_counts.iloc[1] == 1
    assert value_counts.iloc[2] == 2
    assert value_counts.iloc[3] == 0
    assert value_counts.iloc[8] == 1
