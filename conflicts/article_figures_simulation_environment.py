import copy
from collections import OrderedDict

import numpy as np

from conflicts.simulate import Aircraft, Flow, AircraftInFlow, CombinedFlows, Simulation


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

def flows_on_collision():
    flow1_kwargs = {
        'position': np.array([[0, 20], [0, 15], [0, 10], [0, 5]], dtype=Aircraft.dtype),
        'trk': np.array([90, 90, 90, 180], dtype=Aircraft.dtype),
        'gs': np.array([10, 10, 10, 10], dtype=Aircraft.dtype),
        'alt': np.array([2000, 500, 2000, 2000], dtype=Aircraft.dtype),
        'vs': np.array([0, 0, 0, 0], dtype=Aircraft.dtype),
        'callsign': np.array(['ac1_1', 'ac1_2', 'ac1_3', 'ac1_4']),
        'active': np.array([True, True, True, True], dtype=bool),
    }
    flow2_kwargs = {
        'position': np.array([[10, 20], [10, 10], [10, 15]], dtype=Aircraft.dtype),
        'trk': np.array([270, 270, 270], dtype=Aircraft.dtype),
        'gs': np.array([10, 10, 10], dtype=Aircraft.dtype),
        'alt': np.array([2000, 2000, 2000], dtype=Aircraft.dtype),
        'vs': np.array([0, 0, -23.2], dtype=Aircraft.dtype),
        'callsign': np.array(['ac1_1', 'ac1_2', 'ac1_3']),
        'active': np.array([True, True, True], dtype=bool),
    }
    flow1 = Flow(**flow1_kwargs)
    flow2 = Flow(**flow2_kwargs)
    return flow1, flow2, flow1_kwargs, flow2_kwargs


flow1, flow2, flow1_kwargs, _ = copy.deepcopy(flows_on_collision())
flow3_args = copy.deepcopy(aircraft_on_collision())
flow3 = Flow(*flow3_args[:-2])
flows = OrderedDict()
flows['flow1'] = flow1
flows['flow2'] = flow2
flows['flow3'] = flow3
combined_flows = CombinedFlows(copy.deepcopy(flows))
sim = Simulation(combined_flows, plot_frequency=20,
                 savefig_str='pgf/simulation_environment_example-{:.4f}.pgf')
sim.simulate(20, 0.7, T_conflict_window=[0, 0.2])