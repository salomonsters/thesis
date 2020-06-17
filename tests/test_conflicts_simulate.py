import logging

import numpy as np
import pytest
from pytest import approx
from pint import UnitRegistry, Quantity

from conflicts.simulate import Aircraft, Flow, get_in_unit
ureg = UnitRegistry()

def test_Aircraft():
    ac1 = Aircraft((0, 0), 90, 100, 0, 0, 'ac1')
    assert ac1.v.dtype == ac1.dtype
    assert ac1.position.shape == (2,)
    assert ac1.position.dtype == ac1.dtype

    ac1.step(1)
    assert np.allclose(ac1.position, [100, 0])
    ac1_respawn = Aircraft((0, 0), 90, 100, 0, 0, 'ac1')
    for _ in range(3600):
        ac1_respawn.step(1/3600.)
    assert np.allclose(ac1_respawn.position, [100, 0])

    ac2 = Aircraft((3, 4), np.degrees(0.5*np.pi - np.arctan2(4, 3)), 5, 0, 0, 'ac2')
    ac2.step(1)
    assert np.allclose(ac2.position, [6, 8])

    ac3 = Aircraft((0, 0), 0, 100, 6000, -100, 'ac3')
    assert ac3.vs_fph == approx(-6000)
    ac3.step(1)
    assert ac3.alt == approx(0)
    assert ac3.position == approx((0, 100))
    assert ac3.v == approx((0, 100))
    ac3.trk = 90
    assert ac3.v == approx((100, 0))
    ac3.step(1)
    assert ac3.position == approx((100, 100))
    ac3.gs = 50
    assert ac3.v == approx((50, 0))
    ac3.step(1)
    assert ac3.position == approx((150, 100))

    two_hour = 2*3600*ureg.second
    ac3.step(two_hour)
    assert ac3.position == approx((250, 100))

    with pytest.raises(NotImplementedError):
        ac1 & 3

    assert ac1 & ac1_respawn == True
    assert ac1 & ac2 == False

    assert Aircraft((0, 0), 0, 100, 0, 0, 'ac1') & Aircraft((0, 3.), 0, 100, 0, 0, 'ac1')
    assert Aircraft((0, 0), 0, 100, 0, 0, 'ac1') & Aircraft((0, 3.), 0, 100, 999, 0, 'ac1')
    assert Aircraft((0, 0), 0, 100, 0, 0, 'ac1') & Aircraft((0, 3.), 0, 100, 1000., 0, 'ac1')
    assert not Aircraft((0, 0), 0, 100, 0, 0, 'ac1') & Aircraft((0, 3.), 0, 100, 1001., 0, 'ac1')
    assert not Aircraft((0, 0), 0, 100, 0, 0, 'ac1') & Aircraft((0, 3.1), 0, 100, 0., 0, 'ac1')

    x0 = (1., 2.)
    inactive_ac = Aircraft(x0, 180, 100, 2000, 0, 'inactive', False)
    inactive_ac.step(0.1)
    assert inactive_ac.position == approx(x0)


LOGGER = logging.getLogger(__name__)


def test_Flow(caplog):
    aircraft_list = [Aircraft((0, 0), 90, 100, 0, 0, 'ac1', True),
                     Aircraft((0, 0), 360, 100, 0, 0, 'ac2', False),
                     Aircraft((0, 0), 90, 100, 0, 0, 'ac3', False)]
    flow = Flow(aircraft_list)
    assert flow.position.shape == (3, 2)
    assert flow.v.shape == (3, 2)
    assert flow.alt.shape == (3, )
    assert flow.vs_fph.shape == (3, )
    assert flow.active.shape == (3,)

    assert flow.active == approx(np.array([True, False, False]))
    flow.step(1)
    assert flow.position == approx(np.array([[100, 0], [0, 0], [0, 0]]))

    flow.activate('ac1')
    assert 'ac1 was already active' in caplog.text
    flow.activate('ac1', deactivate=True)

    flow.deactivate('ac2')
    assert 'ac2 was already inactive' in caplog.text
    flow.activate('ac2')

    flow.step(1)
    assert flow.position == approx(np.array([[100, 0], [0, 100], [0, 0]]))
