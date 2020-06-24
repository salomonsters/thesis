import copy
import itertools
import logging
from collections import OrderedDict
from numbers import Number

import matplotlib.pyplot as plt
import numba
import numpy as np
from numpy.linalg import norm
from pint import UnitRegistry, Quantity

ureg = UnitRegistry()


class Aircraft:
    horizontal_separation_requirement = 3  # nm
    vertical_separation_requirement = 1000  # ft
    dtype = np.float
    vx, vy, v = None, None, None

    def __init__(self, position, trk, gs, alt, vs, callsign, active=True):
        if isinstance(position, np.ndarray):
            self.position = position
        else:
            self.position = np.array(position, dtype=self.dtype)  # nm
        self._gs = gs  # knots
        self._trk = trk  # degrees
        self._trk_rad = np.radians(trk)
        self.alt = alt  # feet
        self._vs_fpm = vs  # feet per minute
        self.vs_fph = vs * 60.
        self.callsign = callsign
        self.active = active
        self.calculate_speed_vector()

    def calculate_speed_vector(self):
        self.vx = self._gs * np.sin(self._trk_rad)  # knots
        self.vy = self._gs * np.cos(self._trk_rad)  # knots
        self.v = np.array([self.vx, self.vy])  # knots

    @property
    def gs(self):
        return self._gs

    @gs.setter
    def gs(self, gs):
        raise NotImplementedError()

    @property
    def trk(self):
        return self._trk

    @trk.setter
    def trk(self, trk):
        raise NotImplementedError()

    @property
    def vs(self):
        return self._vs_fpm

    @vs.setter
    def vs(self, vs):
        raise NotImplementedError()

    @staticmethod
    def convert_dt(dt):
        if isinstance(dt, Quantity):
            dt = dt.to(ureg.hour).magnitude
        dt = Aircraft.dtype(dt)
        return dt

    def step(self, dt):
        dt = self.convert_dt(dt)
        if self.active:
            self.position += self.v * dt
            self.alt += self.vs_fph * dt

    def __and__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError()
        return np.linalg.norm(self.position - other.position) <= self.horizontal_separation_requirement and \
               np.abs(self.alt - other.alt) <= self.vertical_separation_requirement


class SingleAircraft(Aircraft):
    @Aircraft.gs.setter
    def gs(self, gs):
        self._gs = gs
        self.calculate_speed_vector()

    @Aircraft.trk.setter
    def trk(self, trk):
        self._trk = trk
        self._trk_rad = np.radians(trk)
        self.calculate_speed_vector()

    @Aircraft.vs.setter
    def vs(self, vs):
        self._vs_fpm = vs
        self.vs_fph = vs * 60.


class AircraftInFlow(Aircraft):
    def __init__(self, index, position, trk, gs, alt, vs, callsign, active: np.ndarray):
        self.index = index
        super().__init__(position[index:index + 1], trk[index:index + 1], gs[index:index + 1], alt[index:index + 1],
                         vs[index:index + 1], callsign[index:index + 1], active[index:index + 1])

    def step(self, *args, **kwargs):
        raise RuntimeError("Step is not implemented for AircraftInFlow")


class Flow:
    callsign_map = None
    t_lookahead = 5./60.

    def __init__(self, position, trk, gs, alt, vs, callsign, active, calculate_collisions=False):
        self.n = self._check_dimensions_and_get_n(position, trk, gs, alt, vs, callsign, active)
        self.position = position
        self.trk = trk
        self.gs = gs
        self.alt = alt
        self.vs = vs
        self.callsign = callsign
        self.active = active
        self.callsign_map = {}
        self._populate_callsign_map(self.callsign)

        self.v = (gs * np.array([np.sin(np.radians(trk)), np.cos(np.radians(trk))])).T
        self.vs_fph = self.vs * 60.

        self.index = np.arange(self.n, dtype=int)
        self.aircraft = [AircraftInFlow(i, self.position, self.trk, self.gs, self.alt, self.vs, self.callsign,
                                        self.active) for i in self.index]
        if calculate_collisions:
            self.collisions = np.empty((self.n, self.n), dtype=bool)
        self.calculate_collisions = calculate_collisions
        self.conflicts = np.empty((self.n, self.n), dtype=bool)
        self.within_flow_active_conflicts = np.zeros(self.n, dtype=bool)
        self.all_conflicts = np.zeros(self.n, dtype=bool)

        self._calculate_active_aircraft_combinations()
        self._update_collisions_and_conflicts()

    @classmethod
    def expand_properties(cls, kwargs: dict, n_from='callsign'):
        flow_kwargs = {}
        n = len(kwargs[n_from])
        for k, v in kwargs.items():
            if isinstance(v, Number):
                if isinstance(v, bool):
                    flow_kwargs[k] = v * np.ones(n, dtype=bool)
                else:
                    flow_kwargs[k] = v * np.ones(n, dtype=Aircraft.dtype)
            elif isinstance(v, tuple):
                flow_kwargs[k] = np.zeros((n, len(v)), dtype=Aircraft.dtype)
                flow_kwargs[k][:, :] = v
            elif isinstance(v, list):
                if len(v) != n:
                    raise ValueError("Incorrect first dimension for key {}: {}"
                                     " (should be {} based on {}".format(k, len(v), n, n_from))
                flow_kwargs[k] = np.array(v)
            elif isinstance(v, np.ndarray):
                if v.shape[0] != n:
                    raise ValueError("Incorrect first dimension for key {}: {}"
                                     " (should be {} based on {}".format(k, v.shape[0], n, n_from))
                flow_kwargs[k] = v
            else:
                raise ValueError("Unsupported type for key {}: {}".format(k, type(v)))
        return cls(**flow_kwargs)

    def _calculate_active_aircraft_combinations(self):
        self.active_aircraft_combinations = list(itertools.combinations_with_replacement(self.index[self.active], 2))

    def activate(self, callsign, deactivate=False):
        if deactivate:
            if not self.active[self.callsign_map[callsign]]:
                logging.warning("Aircraft {} was already inactive".format(callsign))
            self.active[self.callsign_map[callsign]] = False
        else:
            if self.active[self.callsign_map[callsign]]:
                logging.warning("Aircraft {} was already active".format(callsign))
            self.active[self.callsign_map[callsign]] = True
        self._calculate_active_aircraft_combinations()
        self._update_collisions_and_conflicts()

    def deactivate(self, callsign):
        return self.activate(callsign, True)

    def step(self, dt, update_conflicts=True):
        dt = Aircraft.convert_dt(dt)
        self.position[self.active] += self.v[self.active] * dt
        self.alt[self.active] += self.vs_fph[self.active] * dt
        if update_conflicts:
            self._update_collisions_and_conflicts()

    def _update_collisions_and_conflicts(self):
        # Inactive aircraft have no collisions
        if self.calculate_collisions:
            self.collisions[:, :] = False
            for i, j in self.active_aircraft_combinations:
                if i != j:
                    self.collisions[i, j] = self.collisions[j, i] = self.aircraft[i] & self.aircraft[j]

        self.conflicts[:, :] = conflicts_between_multiple(self, t_lookahead=self.t_lookahead)
        self.within_flow_active_conflicts[:] = self.conflicts.sum(axis=1)
        self.all_conflicts[:] = self.all_conflicts | self.within_flow_active_conflicts

    def _populate_callsign_map(self, callsign_list):
        for i, callsign in enumerate(callsign_list):
            if callsign in self.callsign_map.keys():
                raise ValueError("Duplicate callsign detected")
            self.callsign_map[callsign] = i

    @staticmethod
    def _check_dimensions_and_get_n(*args):
        n = args[0].shape[0]
        for arg in args:
            if arg.shape[0] != n:
                raise ValueError('Not all arguments have same first dimension')
        return n

    def __getitem__(self, item):
        if item == 'pos':
            item = 'position'
        try:
            return self.__getattribute__(item)[self.active]
        except IndexError:
            return self.__getattribute__(item)
        except AttributeError:
            raise NotImplementedError('No getitem defined for item {}'.format(item))


class CombinedFlows:
    t_lookahead = 5/60

    def __init__(self, flows: OrderedDict):
        self.flows = copy.deepcopy(flows)
        self.flow_keys = list(self.flows.keys())
        self.flow_key_pairs = list(itertools.combinations(self.flow_keys, 2))
        self.current_conflict_state = {
            (flow_a, flow_b): np.zeros((self.flows[flow_a].n, self.flows[flow_b].n), dtype=bool)
            for flow_a, flow_b in self.flow_key_pairs}
        self.all_conflicts = {
            (flow_a, flow_b): np.zeros((self.flows[flow_a].n, self.flows[flow_b].n), dtype=bool)
            for flow_a, flow_b in self.flow_key_pairs}
        self.active_conflicts_within_flow_or_between_flows = {k: np.zeros(self.flows[k].n, dtype=bool) for k in self.flow_keys}
        self.current_conflict_state.update({k: self.flows[k].conflicts for k in self.flow_keys})
        self._update_conflicts()

    def _update_conflicts(self):
        for flow in self.flow_keys:
            self.active_conflicts_within_flow_or_between_flows[flow][:] = False
        for flow_a, flow_b in self.flow_key_pairs:
            conflicts_between_multiple(self.flows[flow_a], self.flows[flow_b], t_lookahead=self.t_lookahead,
                                       out=self.current_conflict_state[(flow_a, flow_b)])
            self.active_conflicts_within_flow_or_between_flows[flow_a][:] = self.flows[flow_a].within_flow_active_conflicts | self.current_conflict_state[(flow_a, flow_b)].sum(axis=1) | self.active_conflicts_within_flow_or_between_flows[flow_a]
            self.active_conflicts_within_flow_or_between_flows[flow_b][:] = self.flows[flow_b].within_flow_active_conflicts | self.current_conflict_state[(flow_a, flow_b)].sum(axis=0) | self.active_conflicts_within_flow_or_between_flows[flow_b]
            self.all_conflicts[(flow_a, flow_b)][:, :] = self.current_conflict_state[(flow_a, flow_b)] | self.all_conflicts[(flow_a, flow_b)]

    def step(self, dt, update_conflicts=True):
        dt = Aircraft.convert_dt(dt)
        for k in self.flow_keys:
            self.flows[k].step(dt, update_conflicts)
        if update_conflicts:
            self._update_conflicts()

    def __getitem__(self, item):
        return self.flows[item]


def conflict_between(own: Aircraft, intruder: Aircraft, t_lookahead=5./60):
    if not (own.active and intruder.active):
        return False
    x_rel = (intruder.position - own.position).reshape((2,))
    v_rel = (own.v - intruder.v).reshape((2,))

    with np.errstate(invalid='ignore'):
        t_horizontal_conflict = (np.inner(x_rel, v_rel) +
                                 np.array((-1, 1)) * np.sqrt(
                    np.inner(x_rel, v_rel)**2 - norm(x_rel)**2*norm(v_rel)**2+norm(v_rel)**2
                    * Aircraft.horizontal_separation_requirement**2)
                                 )/(norm(v_rel)**2)
        t_cpa = np.sum(t_horizontal_conflict)/2.
        minimum_distance = norm(t_cpa*v_rel - x_rel)
        t_in_hor, t_out_hor = min(t_horizontal_conflict), max(t_horizontal_conflict)
        horizontal_conflict = minimum_distance < Aircraft.horizontal_separation_requirement \
                              and t_out_hor > 0\
                              and t_in_hor < t_lookahead
        if not horizontal_conflict:
            return False

        vs_rel_fph = own.vs_fph - intruder.vs_fph
        alt_diff = intruder.alt - own.alt

        if np.abs(vs_rel_fph) < 1e-8:
            if np.abs(alt_diff) < Aircraft.vertical_separation_requirement:
                # We have a horizontal conflict and we are flying level at conflicting altitudes, so we have a conflict
                return True

        t_vertical_conflict = (intruder.alt - own.alt + np.array((-1, 1))*Aircraft.vertical_separation_requirement
                               )/vs_rel_fph
        t_in_vert, t_out_vert = min(t_vertical_conflict), max(t_vertical_conflict)

        t_in_combined = max(t_in_hor, t_in_vert)

        if t_in_combined < t_lookahead:
            return True
        return False


@numba.jit(parallel=True, error_model='numpy')
def calculate_horizontal_conflict(shape, own_position, other_position, own_v, other_v, separation_requirement_sq,
                                  t_in_hor, t_out_hor, minimum_distance):
    for i in numba.prange(shape[0]):
        for j in numba.prange(shape[1]):
            position_diff_x = -(own_position[i][0] - other_position[j][0])
            position_diff_y = -(own_position[i][1] - other_position[j][1])
            v_diff_x = own_v[i][0] - other_v[j][0]
            v_diff_y = own_v[i][1] - other_v[j][1]
            x_v_inner = position_diff_x * v_diff_x + position_diff_y * v_diff_y
            x_rel_norm_sq = position_diff_x * position_diff_x + position_diff_y * position_diff_y
            v_rel_norm_sq = v_diff_x * v_diff_x + v_diff_y * v_diff_y
            t_horizontal_conflict_1 = (x_v_inner + np.sqrt(
                x_v_inner ** 2 - x_rel_norm_sq * v_rel_norm_sq + v_rel_norm_sq * separation_requirement_sq)) / v_rel_norm_sq
            t_horizontal_conflict_2 = (x_v_inner - np.sqrt(
                x_v_inner ** 2 - x_rel_norm_sq * v_rel_norm_sq + v_rel_norm_sq * separation_requirement_sq)) / v_rel_norm_sq
            t_cpa = 0.5 * (t_horizontal_conflict_1 + t_horizontal_conflict_2)
            minimum_distance[i, j] = t_cpa * (-position_diff_x * v_diff_x - position_diff_y * v_diff_y)
            if t_horizontal_conflict_2 > t_horizontal_conflict_1:
                t_in_hor[i, j] = t_horizontal_conflict_1
                t_out_hor[i, j] = t_horizontal_conflict_2
            else:
                t_in_hor[i, j] = t_horizontal_conflict_2
                t_out_hor[i, j] = t_horizontal_conflict_1


def conflicts_between_multiple(own: Flow, other: Flow=None, t_lookahead=5. / 60, out=None):
    if other is None:
        other = own
    t_in_hor = np.zeros((own.n, other.n), dtype=Aircraft.dtype)
    t_out_hor = np.zeros((own.n, other.n), dtype=Aircraft.dtype)
    minimum_distance = np.zeros((own.n, other.n), dtype=Aircraft.dtype)
    calculate_horizontal_conflict((own.n, other.n), own.position, other.position, own.v, other.v,
                                  Aircraft.horizontal_separation_requirement ** 2,
                                  t_in_hor, t_out_hor, minimum_distance)
    with np.errstate(invalid='ignore'):
        horizontal_conflict = (minimum_distance < Aircraft.horizontal_separation_requirement) \
                              & (t_out_hor > 0)\
                              & (t_in_hor < t_lookahead)

    both_active = np.expand_dims(own.active, 1) & np.expand_dims(other.active, 0)

    vs_rel_fph = -(np.expand_dims(own.vs_fph, 1) - np.expand_dims(other.vs_fph, 0))
    alt_diff = np.expand_dims(own.alt, 1) - np.expand_dims(other.alt, 0)
    with np.errstate(divide='ignore'):
        t_vertical_conflict = np.array([(alt_diff + Aircraft.vertical_separation_requirement)/vs_rel_fph,
                                        (alt_diff - Aircraft.vertical_separation_requirement)/vs_rel_fph])
        t_in_vert = np.min(t_vertical_conflict, axis=0)
        t_out_vert = np.max(t_vertical_conflict, axis=0)
    with np.errstate(invalid='ignore'):
        t_in_combined = np.max(np.array([t_in_hor, t_in_vert]), axis=0)
        vertical_conflict = (t_in_combined < t_lookahead) & (t_out_vert > 0)
    level_conflict = ((np.abs(vs_rel_fph) < 1e-8) & (np.abs(alt_diff) < Aircraft.vertical_separation_requirement))
    if out is None:
        return horizontal_conflict & (level_conflict | vertical_conflict) & both_active
    else:
        out[:, :] = horizontal_conflict & (level_conflict | vertical_conflict) & both_active

#
#
# import numba
#
#
# @numba.njit(error_model='numpy', debug=True)
# def conflict_between_numba(own, intruder, t_lookahead=5. / 60):
#     both_active = (own.active & intruder.active)
#     x_rel = (intruder.position - own.position).reshape((2,))
#     v_rel = (own.v - intruder.v).reshape((2,))
#
#     t_horizontal_conflict = (np.dot(x_rel, v_rel) +
#                              np.array((-1, 1)) * np.sqrt(
#                 np.dot(x_rel, v_rel) ** 2 - norm(x_rel) ** 2 * norm(v_rel) ** 2 + norm(v_rel) ** 2
#                 * own.horizontal_separation_requirement ** 2)
#                              ) / (norm(v_rel) ** 2)
#     t_cpa = np.sum(t_horizontal_conflict) / 2.
#     minimum_distance = norm(t_cpa * v_rel - x_rel)
#     t_in_hor, t_out_hor = min(t_horizontal_conflict), max(t_horizontal_conflict)
#     horizontal_conflict = (minimum_distance < own.horizontal_separation_requirement) \
#                           & (t_out_hor > 0) \
#                           & (t_in_hor < t_lookahead)
#
#     vs_rel_fph = own.vs_fph - intruder.vs_fph
#     alt_diff = intruder.alt - own.alt
#
#     # We have a horizontal conflict and we are flying level at conflicting altitudes, so we have a conflict
#     level_conflict = (np.abs(vs_rel_fph) < 1e-8) & (np.abs(alt_diff) < own.vertical_separation_requirement)
#
#     t_vertical_conflict = (intruder.alt - own.alt + np.array((-1, 1)) * own.vertical_separation_requirement
#                            ) / vs_rel_fph
#     t_in_vert, t_out_vert = min(t_vertical_conflict), max(t_vertical_conflict)
#
#     t_in_combined = max(t_in_hor, t_in_vert)
#     vertical_conflict = (t_in_combined < t_lookahead) & (t_out_vert > 0)
#
#     return horizontal_conflict & (level_conflict | vertical_conflict) & both_active


class Simulation:
    t = 0.
    i = 0
    T = 0.

    def __init__(self, flows: CombinedFlows, activators=None, plot_frequency=None):
        self.flows = flows
        self.activators = activators
        if activators is not None:
            self.activators_iter = iter(activators.T)
        else:
            self.activators_iter = None
        self.plot_frequency = plot_frequency

    def simulate(self, f, T, conflict_frequency=1):
        dt = 1/f
        self.T = T
        if self.plot_frequency is not None:
            relative_plot_frequency = f // self.plot_frequency
            self.prepare_plot()
        while self.t + dt < T:
            self.fire_activators()
            self.flows.step(dt, update_conflicts=self.i % conflict_frequency == 0)

            if self.plot_frequency is not None and self.i % relative_plot_frequency == 0:
                self.plot_in_loop()

            self.t += dt
            self.i += 1

    def fire_activators(self):
        if self.activators is not None:
            fire = next(self.activators_iter)
            if np.any(fire):
                for i, flow in enumerate(self.flows.flow_keys):
                    if fire[i] and len(np.where(~self.flows[flow].active)[0]) > 0:
                        callsign = self.flows[flow].callsign[np.where(~self.flows[flow].active)[0][0].item()]
                        self.flows[flow].activate(callsign)

    def prepare_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion()
        self.xlim = [-11, 20]
        self.ylim = [-11, 20]
        x_axis_scale_in_units = np.diff(self.xlim).item()
        y_axis_scale_in_units = np.diff(self.ylim).item()
        bbox = self.ax.get_window_extent().transformed(self.fig.dpi_scale_trans.inverted())
        width, height = bbox.width, bbox.height
        width *= x_axis_scale_in_units / self.fig.dpi
        height *= y_axis_scale_in_units / self.fig.dpi

        self.lw_no_conflict = 1 / width * x_axis_scale_in_units
        self.lw_conflict = 3 / width * x_axis_scale_in_units

    def plot_in_loop(self):
        plt.clf()
        for flow_i, flow in enumerate(self.flows.flow_keys):
            active_conflicts = self.flows.active_conflicts_within_flow_or_between_flows[flow][self.flows[flow].active]
            plt.plot(self.flows[flow]['position'][active_conflicts][:, 0],
                     self.flows[flow]['position'][active_conflicts][:, 1], lw=0, marker='o', fillstyle='none',
                     c='C{}'.format(flow_i), markersize=self.lw_conflict)
            plt.plot(self.flows[flow]['position'][~active_conflicts][:, 0],
                     self.flows[flow]['position'][~active_conflicts][:, 1], lw=0, marker='o', fillstyle='none',
                     c='C{}'.format(flow_i), markersize=self.lw_no_conflict, label=flow)
            plt.quiver(self.flows[flow]['position'][:, 0], self.flows[flow]['position'][:, 1],
                       self.flows[flow]['v'][:, 0], self.flows[flow]['v'][:, 1], color='C{}'.format(flow_i),
                       angles='xy', scale_units='xy', scale=1 / Flow.t_lookahead, width=0.001)

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        plt.legend()
        if self.t > 0:
            progress = self.t / self.T
            progress_divider = 1/self.lw_conflict
            title = int(progress // progress_divider) * '#' +int((1 - progress) // progress_divider )* '_'
            plt.title(title)
        plt.pause(0.05)

    @property
    def aggregated_conflicts(self):
        aggregated_conflicts = {k: np.sum(self.flows[k].all_conflicts) for k in self.flows.flow_keys}
        aggregated_conflicts.update({k: np.sum(self.flows.all_conflicts[k]) for k in self.flows.flow_key_pairs})
        return aggregated_conflicts

if __name__ == "__main__":
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
    pos = np.array([[0, 0], [10, 0], [5, 10], [5, 5]], dtype=Aircraft.dtype)
    trk = np.array([90, 270, 180, 0], dtype=Aircraft.dtype)
    gs = np.array([10, 10, 10, 5], dtype=Aircraft.dtype)
    alt = np.array([2000, 1000, 2000, 2000], dtype=Aircraft.dtype)
    vs = np.array([0, 1900 / 30., 0, 0], dtype=Aircraft.dtype)
    callsign = np.array(['ac1', 'ac2', 'ac3', 'ac4'])
    active = np.array([True, True, True, True], dtype=bool)
    index = np.arange(4, dtype=int)
    ac = [AircraftInFlow(i, pos, trk, gs, alt, vs, callsign, active) for i in index]
    flow3_args = copy.deepcopy((pos, trk, gs, alt, vs, callsign, active, index, ac))
    flow3 = Flow(*flow3_args[:-2])
    flows = OrderedDict()
    flows['flow1'] = flow1
    flows['flow2'] = flow2
    flows['flow3'] = flow3
    combined_flows = CombinedFlows(copy.deepcopy(flows))
    current_conflict_state = combined_flows.current_conflict_state
    sim = Simulation(combined_flows, plot_frequency=20)
    sim.simulate(20, 0.5)



    if input("Continue?") == "Y":
        flow0_kwargs = {
            'position': (-10, 0),
            'trk': 120,
            'gs': 80,
            'alt': 2000,
            'vs': 0,
            'callsign': ['flow_{0}_ac_{1}'.format(0, i) for i in range(100)],
            'active': False,
        }
        flow1_kwargs = {
            'position': (0, -10),
            'trk': 45,
            'gs': 80,
            'alt': 1500,
            'vs': 0,
            'callsign': ['flow_{0}_ac_{1}'.format(1, i) for i in range(101)],
            'active': False,
        }
        flow1 = Flow.expand_properties(flow0_kwargs)
        flow2 = Flow.expand_properties(flow1_kwargs)

        flows = CombinedFlows(OrderedDict([('flow1', flow1), ('flow2', flow2)]))
        rg = np.random.default_rng()
        f_simulation = 20 * 3600
        f_plot = 3600 // 30
        T = 1 # hr
        activators = np.array([rg.random(f_simulation) < 100/f_simulation,
                               rg.random(f_simulation) < 100/f_simulation], dtype=bool)
        activators[:, 0] = True
        sim = Simulation(flows, activators, plot_frequency=f_simulation//20)
        # sim.simulate(f_simulation, T, conflict_frequency=f_simulation//20)
        sim.simulate(f_simulation, T)


    # t = 0.
    # dt = 5 / 3600.
    # i = 0
    #
    #
    # while t < 5:
    #
    #     if i % 100 == 0:
    #         next_callsign = flows['flow1'].callsign[~flows['flow1'].active][0]
    #         flows['flow1'].activate(next_callsign)
    #
    #     elif i % 101 == 0:
    #         next_callsign = flows['flow2'].callsign[~flows['flow2'].active][0]
    #         flows['flow2'].activate(next_callsign)
    #     if i % 10 == 0:
    #
    #
    #     flows.step(dt)
    #
    #     i += 1
    #     t += dt
