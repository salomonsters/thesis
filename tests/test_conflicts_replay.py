import copy
from collections import OrderedDict

import numpy as np
import pandas as pd
import pytest

from conflicts.replay import ReplayFlow, ReplaySimulation
from conflicts.simulate import CombinedFlows

f_simulation = 60


@pytest.fixture
def FlowWithOvertakes():
    T = 1800  # seconds
    gs0 = 100  # knots
    gs1 = 150
    alt = 2000
    vs = 0
    y0 = 0
    x0_0 = 0
    x0_1 = 15 * 1852
    trk = 270
    ts = np.arange(0, T, f_simulation)
    df0 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs0,
                        'trk': np.ones_like(ts) * trk,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0_0 + ts * gs0 * np.sin(np.radians(trk)) * 1852 / 3600,
                        'y': y0 + ts * gs0 * np.cos(np.radians(trk)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac0'])
                        })
    df1 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs1,
                        'trk': np.ones_like(ts) * trk,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0_1 + ts * gs1 * np.sin(np.radians(trk)) * 1852 / 3600,
                        'y': y0 + ts * gs1 * np.cos(np.radians(trk)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac1'])
                        })
    df = pd.concat([df0, df1])
    return df


@pytest.fixture
def OrthogonalFlow():
    T = 1800  # seconds
    gs0 = 150  # knots
    gs1 = 130
    gs2 = 100
    alt = 2000
    vs = 0
    y0 = 15 * 1852
    x0 = 0
    y0_2 = y0 + 6 * 1852
    x0_3 = -5 * 1852
    y0_3 = -5 * 1852
    trk01 = 180
    trk2 = 0
    ts = np.arange(0, T, f_simulation)
    df0 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs0,
                        'trk': np.ones_like(ts) * trk01,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0 + ts * gs0 * np.sin(np.radians(trk01)) * 1852 / 3600,
                        'y': y0 + ts * gs0 * np.cos(np.radians(trk01)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac0'])
                        })
    df1 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs1,
                        'trk': np.ones_like(ts) * trk01,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0 + ts * gs1 * np.sin(np.radians(trk01)) * 1852 / 3600,
                        'y': y0 + ts * gs1 * np.cos(np.radians(trk01)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac1'])
                        })
    df2 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs2,
                        'trk': np.ones_like(ts) * trk2,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0 + ts * gs1 * np.sin(np.radians(trk2)) * 1852 / 3600,
                        'y': y0_2 + ts * gs1 * np.cos(np.radians(trk2)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac2'])
                        })
    df3 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs2,
                        'trk': np.ones_like(ts) * trk2,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0_3 + ts * gs1 * np.sin(np.radians(trk2)) * 1852 / 3600,
                        'y': y0_3 + ts * gs1 * np.cos(np.radians(trk2)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac3'])
                        })
    df3.loc[15:, 'x'] = 0
    df3.loc[15:, 'y'] += 20 * 1852
    df3.loc[15:, 'gs'] = 200

    df = pd.concat([df0, df1, df2, df3])
    return df


@pytest.fixture
def MultiConflictInFlow():
    T = 1800  # seconds
    gs0 = 150  # knots
    gs1 = 130
    gs2 = 100
    alt = 2000
    vs = 0
    y0 = 15 * 1852
    x0 = 0
    x0_2 = 0.5 * 1852
    y0_2 = -30 * 1852
    trk01 = 180
    trk2 = 0
    ts = np.arange(0, T, f_simulation)
    df0 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs0,
                        'trk': np.ones_like(ts) * trk01,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0 + ts * gs0 * np.sin(np.radians(trk01)) * 1852 / 3600,
                        'y': y0 + ts * gs0 * np.cos(np.radians(trk01)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac0'])
                        })
    df1 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs1,
                        'trk': np.ones_like(ts) * trk01,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0 + ts * gs1 * np.sin(np.radians(trk01)) * 1852 / 3600,
                        'y': y0 + ts * gs1 * np.cos(np.radians(trk01)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac1'])
                        })
    df2 = pd.DataFrame({'ts': ts,
                        'gs': np.ones_like(ts) * gs2,
                        'trk': np.ones_like(ts) * trk2,
                        'alt': np.ones_like(ts) * alt,
                        'roc': np.ones_like(ts) * vs,
                        'x': x0_2 + ts * gs1 * np.sin(np.radians(trk2)) * 1852 / 3600,
                        'y': y0_2 + ts * gs1 * np.cos(np.radians(trk2)) * 1852 / 3600,
                        'callsign': np.array(len(ts) * ['ac2'])
                        })
    df = pd.concat([df0, df1, df2])
    return df


def test_conflicts_within_flow(FlowWithOvertakes):
    flow = ReplayFlow(FlowWithOvertakes, 'callsign', '0')
    flows_dict = OrderedDict({'0': flow})
    flows = CombinedFlows(flows_dict)
    flows_dict_copy = copy.deepcopy(flows_dict)
    sim = ReplaySimulation(flows, plot_frequency=None, calculate_conflict_per_time_unit=True,
                           replay_df=FlowWithOvertakes)
    assert np.all(flow.active == False)
    sim.simulate(f_simulation, conflict_frequency=f_simulation, T=0.24)
    assert np.all(sim.flows.active_conflicts_within_flow_or_between_flows['0'] == [True, True])
    sim.simulate(f_simulation, conflict_frequency=f_simulation, T=0.4)
    assert np.all(flow.active == False)

    assert sim.aggregated_conflicts == pytest.approx({'0': 1})

    flows_copy = CombinedFlows(flows_dict_copy)

    flows_copy.step(0.01, t=0.01)
    assert flows_copy.current_conflict_state['0'] == pytest.approx(np.array([[False, False], [False, False]]))
    assert flows_copy.active_conflicts_within_flow_or_between_flows['0'] == pytest.approx(np.array([False, False]))
    flows_copy.step(0.24, t=0.25)
    assert flows_copy.current_conflict_state['0'] == pytest.approx(np.array([[False, True], [True, False]]))
    assert flows_copy.active_conflicts_within_flow_or_between_flows['0'] == pytest.approx(np.array([True, True]))
    flows_copy.step(0.30, t=0.75)
    assert flows_copy.current_conflict_state['0'] == pytest.approx(np.array([[False, False], [False, False]]))


def test_conflicts_between_flows(FlowWithOvertakes, OrthogonalFlow):
    flow0 = ReplayFlow(FlowWithOvertakes, 'callsign', '0')
    flow1 = ReplayFlow(OrthogonalFlow, 'callsign', '1')
    flows_dict = OrderedDict({'0': flow0, '1': flow1})
    flows = CombinedFlows(flows_dict)
    df_combined = pd.concat([FlowWithOvertakes, OrthogonalFlow])
    # flows_dict_copy = copy.deepcopy(flows_dict)
    sim = ReplaySimulation(flows, plot_frequency=f_simulation, calculate_conflict_per_time_unit=True, replay_df=df_combined)
    sim.simulate(f_simulation, conflict_frequency=f_simulation, T=0.05)
    assert np.all(sim.flows.active_conflicts_within_flow_or_between_flows['0'] == [True, True])
    assert np.all(sim.flows.active_conflicts_within_flow_or_between_flows['1'] == [True, True, False, True])

    assert sim.flows.current_conflict_state[('0', '1')] == pytest.approx(np.array([[False, False, False, True],
                                                                                   [True, True, False, False]]))
    assert sim.flows.current_conflict_state['0'] == pytest.approx(np.zeros((2, 2), dtype=bool))
    assert sim.flows.current_conflict_state['1'] == pytest.approx(np.array([[False, True, False, False],
                                                                            [True, False, False, False],
                                                                            [False, False, False, False],
                                                                            [False, False, False, False]]))
    sim.simulate(f_simulation, conflict_frequency=f_simulation, T=0.1)
    assert sim.flows.current_conflict_state[('0', '1')] == pytest.approx(np.array([[False, False, False, False],
                                                                                   [True, True, False, False]]))
    sim.simulate(f_simulation, conflict_frequency=f_simulation, T=0.4)
    # assert np.all(flow.active == False)
    # the number 2 is the total number of aircraft in that flow that ever saw a conflict within that flow
    assert sim.flows['0'].conflict_now_or_in_past == pytest.approx(np.array([[False, True], [True, False]]))
    assert sim.flows['1'].conflict_now_or_in_past == pytest.approx(np.array([[False, True, False, False],
                                                                             [True, False, False, False],
                                                                             [False, False, False, True],
                                                                             [False, False, True, False]]))
    assert sim.aggregated_conflicts['0'] == 1
    assert sim.aggregated_conflicts['1'] == 2
    assert sim.aggregated_conflicts[('0', '1')] == 3


def test_multi_conflict_within_flow(MultiConflictInFlow):
    flow = ReplayFlow(MultiConflictInFlow, 'callsign', '0')
    flows_dict = OrderedDict({'0': flow})
    flows = CombinedFlows(flows_dict)
    sim = ReplaySimulation(flows, plot_frequency=None, calculate_conflict_per_time_unit=True,
                           replay_df=MultiConflictInFlow)
    sim.simulate(f_simulation, conflict_frequency=f_simulation, T=0.5)
    assert sim.flows['0'].conflict_now_or_in_past == pytest.approx(np.array([[False, True, True],
                                                                             [True, False, True],
                                                                             [True, True, False]]))
    assert sim.aggregated_conflicts == pytest.approx({'0': 3})
