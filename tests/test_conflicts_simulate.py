import copy
import logging

import numpy as np
import pytest
from pint import UnitRegistry
from pytest import approx

from conflicts.simulate import Aircraft, SingleAircraft, Flow, AircraftInFlow
from conflicts.simulate import conflict_between, conflicts_between_multiple

ureg = UnitRegistry()


def test_aircraft():
    ac = Aircraft((0, 0), 90, 100, 2000, -100, 'ac', True)
    with pytest.raises(NotImplementedError):
        ac.trk = 0
    with pytest.raises(NotImplementedError):
        ac.gs = 0
    with pytest.raises(NotImplementedError):
        ac.vs = 0


def test_single_aircraft():
    ac1 = SingleAircraft((0, 0), 90, 100, 0, 0, 'ac1')
    assert ac1.v.dtype == ac1.dtype
    assert ac1.position.shape == (2,)
    assert ac1.position.dtype == ac1.dtype

    ac1.step(1)
    assert np.allclose(ac1.position, [100, 0])
    ac1_respawn = SingleAircraft((0, 0), 90, 100, 0, 0, 'ac1')
    for _ in range(3600):
        ac1_respawn.step(1 / 3600.)
    assert np.allclose(ac1_respawn.position, [100, 0])

    ac2 = SingleAircraft((3, 4), np.degrees(0.5 * np.pi - np.arctan2(4, 3)), 5, 0, 0, 'ac2')
    ac2.step(1)
    assert np.allclose(ac2.position, [6, 8])

    ac3 = SingleAircraft((0, 0), 0, 100, 6000, -100, 'ac3')
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

    two_hour = 2 * 3600 * ureg.second
    ac3.step(two_hour)
    assert ac3.position == approx((250, 100))

    with pytest.raises(NotImplementedError):
        ac1 & 3

    assert ac1 & ac1_respawn
    assert not ac1 & ac2

    assert SingleAircraft((0, 0), 0, 100, 0, 0, 'ac1') & SingleAircraft((0, 3.), 0, 100, 0, 0, 'ac1')
    assert SingleAircraft((0, 0), 0, 100, 0, 0, 'ac1') & SingleAircraft((0, 3.), 0, 100, 999, 0, 'ac1')
    assert SingleAircraft((0, 0), 0, 100, 0, 0, 'ac1') & SingleAircraft((0, 3.), 0, 100, 1000., 0, 'ac1')
    assert not SingleAircraft((0, 0), 0, 100, 0, 0, 'ac1') & SingleAircraft((0, 3.), 0, 100, 1001., 0, 'ac1')
    assert not SingleAircraft((0, 0), 0, 100, 0, 0, 'ac1') & SingleAircraft((0, 3.1), 0, 100, 0., 0, 'ac1')

    x0 = (1., 2.)
    inactive_ac = SingleAircraft(x0, 180, 100, 2000, 0, 'inactive', False)
    inactive_ac.step(0.1)
    assert inactive_ac.position == approx(x0)


LOGGER = logging.getLogger(__name__)


@pytest.fixture
def aircraft_in_flow_list():
    pos = np.array([[0, 0], [1, 2], [10, 20]], dtype=Aircraft.dtype)
    trk = np.array([0, 90, 270], dtype=Aircraft.dtype)
    gs = np.array([100, 50, 150], dtype=Aircraft.dtype)
    alt = np.array([0, 2000, 3000], dtype=Aircraft.dtype)
    vs = np.array([100, 0, -100], dtype=Aircraft.dtype)
    callsign = np.array(['ac1', 'ac2', 'ac3'])
    active = np.array([True, True, False], dtype=bool)
    index = np.arange(3, dtype=int)
    ac = [AircraftInFlow(i, pos, trk, gs, alt, vs, callsign, active) for i in index]
    return pos, trk, gs, alt, vs, callsign, active, index, ac


def test_aircraft_in_flow(aircraft_in_flow_list):
    pos, trk, gs, alt, vs, callsign, active, index, ac = aircraft_in_flow_list

    with pytest.raises(RuntimeError):
        ac[0].step(1)

    for i in index:
        assert ac[i].position == approx(pos[i:i + 1])
        assert ac[i].trk == approx(trk[i])
        assert ac[i].gs == approx(gs[i])
        assert ac[i].alt == approx(alt[i])
        assert ac[i].vs == approx(vs[i])
        assert ac[i].callsign == callsign[i]
        assert ac[i].active == active[i]


def test_flow(caplog, aircraft_in_flow_list):
    pos, trk, gs, alt, vs, callsign, active, index, ac = aircraft_in_flow_list

    flow = Flow(pos, trk, gs, alt, vs, callsign, active)
    assert flow.position.shape == (3, 2)
    assert flow.v.shape == (3, 2)
    assert flow.alt.shape == (3,)
    assert flow.vs_fph.shape == (3,)
    assert flow.active.shape == (3,)

    assert flow.active == approx(np.array([True, True, False]))

    flow.step(1)
    v_expected = [[0, 100], [50, 0], [-150, 0]]
    pos_expected = [[0, 100], [51, 2], [10, 20]]
    alt_expected = [6000, 2000, 3000]
    for i in index:
        assert flow.v[i] == approx(v_expected[i])
        assert flow.alt[i] == approx(alt_expected[i])
        assert flow.position[i:i + 1] == approx(np.atleast_2d(pos_expected[i]))

        assert ac[i].position == approx(pos[i:i + 1])
        assert ac[i].trk == approx(trk[i])
        assert ac[i].gs == approx(gs[i])
        assert ac[i].alt == approx(alt[i])
        assert ac[i].vs == approx(vs[i])
        assert ac[i].callsign == callsign[i]
        assert ac[i].active == active[i]

    flow.activate('ac1')
    assert 'ac1 was already active' in caplog.text
    flow.activate('ac1', deactivate=True)

    flow.deactivate('ac3')
    assert 'ac3 was already inactive' in caplog.text
    flow.activate('ac3')

    flow.step(1)
    assert flow.position == approx(np.array([[0, 100], [101, 2], [-140, 20]]))


def test_flow_no_duplicate_callsigns(aircraft_in_flow_list):
    pos, trk, gs, alt, vs, callsign, active, index, ac = aircraft_in_flow_list
    with pytest.raises(ValueError):
        callsign = np.array(['ac1', 'ac2', 'ac2'])
        Flow(pos, trk, gs, alt, vs, callsign, active)


@pytest.fixture
def aircraft_on_collision():
    pos = np.array([[0, 0], [10, 0], [5, 10], [5, 5]], dtype=Aircraft.dtype)
    trk = np.array([90, 270, 180, 0], dtype=Aircraft.dtype)
    gs = np.array([10, 10, 10, 5], dtype=Aircraft.dtype)
    alt = np.array([2000, 1000, 2000, 2000], dtype=Aircraft.dtype)
    vs = np.array([0, 1900 / 30., 0, 0], dtype=Aircraft.dtype)
    callsign = np.array(['ac1', 'ac2', 'ac3', 'ac4'])
    active = np.array([True, True, True, True], dtype=bool)
    index = np.arange(4, dtype=int)
    ac = [AircraftInFlow(i, pos, trk, gs, alt, vs, callsign, active) for i in index]
    return pos, trk, gs, alt, vs, callsign, active, index, ac


def test_flow_collision(aircraft_on_collision):

    pos, trk, gs, alt, vs, callsign, active, index, ac = copy.deepcopy(aircraft_on_collision)
    flow = Flow(pos, trk, gs, alt, vs, callsign, active)
    assert ~np.any(flow.collisions)
    flow.step(1.9 / 15)
    assert ~np.any(flow.collisions)
    flow.step(0.2 / 15)
    assert flow.collisions[2, 3] and flow.collisions[3, 2]
    flow.deactivate('ac4')
    assert ~np.any(flow.collisions)
    pos, trk, gs, alt, vs, callsign, active, index, ac = copy.deepcopy(aircraft_on_collision)
    active[3] = False
    flow = Flow(pos, trk, gs, alt, vs, callsign, active)
    flow.step(0.5)
    assert flow.collisions[0, 1] and flow.collisions[1, 0]
    assert np.sum(flow.collisions) == 2
    flow.step(0.1)
    assert np.sum(flow.collisions) == 0


def test_conflict_between(aircraft_on_collision):
    pos, trk, gs, alt, vs, callsign, active, index, ac = copy.deepcopy(aircraft_on_collision)
    flow = Flow(pos, trk, gs, alt, vs, callsign, active)

    flow.deactivate('ac4')
    assert not conflict_between(flow.aircraft[2], flow.aircraft[3], t_lookahead=1)
    flow.activate('ac4')
    assert conflict_between(flow.aircraft[2], flow.aircraft[3], t_lookahead=1)

    assert not conflict_between(flow.aircraft[0], flow.aircraft[1])
    assert conflict_between(flow.aircraft[0], flow.aircraft[1], t_lookahead=.36)
    assert not conflict_between(flow.aircraft[2], flow.aircraft[3])
    flow.step(0.35)
    assert conflict_between(flow.aircraft[0], flow.aircraft[1])
    flow.step(0.10)
    assert conflict_between(flow.aircraft[2], flow.aircraft[3])
    flow.step(0.05)
    assert conflict_between(flow.aircraft[0], flow.aircraft[1])
    flow.step(0.16)
    assert not conflict_between(flow.aircraft[0], flow.aircraft[1])


def test_conflicts_between_multiple(aircraft_on_collision):
    for i in range(2):
        pos, trk, gs, alt, vs, callsign, active, index, ac = copy.deepcopy(aircraft_on_collision)
        flow = Flow(pos, trk, gs, alt, vs, callsign, active)
        if i == 1:
            flow.deactivate('ac4')
        for step, t_lookahead in zip([0.0, 0.35, 0.1, 0.05, 0.16, 0.0],
                                     [5/60, .36, 5/60, 5/60, 5/60, 0.]):
            individual_conflicts = np.array([[conflict_between(flow.aircraft[i], flow.aircraft[j], t_lookahead) for i in flow.index] for j in flow.index])
            combined_conflicts = conflicts_between_multiple(flow, t_lookahead=t_lookahead)
            assert np.alltrue(conflicts_between_multiple(flow, t_lookahead=t_lookahead) ==
                              conflicts_between_multiple(flow, copy.deepcopy(flow), t_lookahead=t_lookahead))
            flow.t_lookahead = t_lookahead
            flow._update_collisions_and_conflicts()
            assert np.alltrue(combined_conflicts == individual_conflicts)
            assert np.alltrue(flow.conflicts == combined_conflicts)
            flow.step(step)


@pytest.fixture
def flows_on_collision():
    flow1_kwargs = {
        'position': np.array([[0, 0], [5, 0]], dtype=Aircraft.dtype),
        'trk': np.array([90, 90], dtype=Aircraft.dtype),
        'gs': np.array([10, 10], dtype=Aircraft.dtype),
        'alt': np.array([500, 2000], dtype=Aircraft.dtype),
        'vs': np.array([0, 0], dtype=Aircraft.dtype),
        'callsign': np.array(['ac1', 'ac2']),
        'active': np.array([True, True], dtype=bool),
    }
    flow2_kwargs = {
        'position': np.array([[10, 5], [10, -5], [10, -15]], dtype=Aircraft.dtype),
        'trk': np.array([180, 0, 0], dtype=Aircraft.dtype),
        'gs': np.array([10, 10, 10], dtype=Aircraft.dtype),
        'alt': np.array([2000, 2000, 1000], dtype=Aircraft.dtype),
        'vs': np.array([0, 0, 0], dtype=Aircraft.dtype),
        'callsign': np.array(['ac3', 'ac4', 'ac5']),
        'active': np.array([True, True, True], dtype=bool),
    }
    flow1 = Flow(**flow1_kwargs)
    flow2 = Flow(**flow2_kwargs)
    return flow1, flow2, flow1_kwargs, flow2_kwargs


def test_flows_on_collision(flows_on_collision):
    flow1, flow2, flow1_kwargs, _ = copy.deepcopy(flows_on_collision)
    assert ~np.any(conflicts_between_multiple(flow1, flow2, t_lookahead=5/60.))
    off_diagonal_indexer = np.where(~np.eye(flow1_kwargs['position'].shape[0],dtype=bool))
    assert np.all(conflicts_between_multiple(flow2, flow2, t_lookahead=1)[off_diagonal_indexer])
    # todo: finish
    # assert np.alltrue(conflicts_between_multiple(flow1, flow2, t_lookahead=0.5) == np.array([[]]))