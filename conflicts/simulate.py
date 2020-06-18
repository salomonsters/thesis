import copy
import itertools
import logging
from collections import OrderedDict

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

    def __init__(self, position, trk, gs, alt, vs, callsign, active):
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

        self.collisions = np.empty((self.n, self.n), dtype=bool)
        self.conflicts = np.empty((self.n, self.n), dtype=bool)

        self._calculate_active_aircraft_combinations()
        self._update_collisions_and_conflicts()

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

    def step(self, dt):
        dt = Aircraft.convert_dt(dt)
        self.position[self.active] += self.v[self.active] * dt
        self.alt[self.active] += self.vs_fph[self.active] * dt
        self._update_collisions_and_conflicts()

    def _update_collisions_and_conflicts(self):
        # Inactive aircraft have no collisions
        self.collisions[:, :] = False
        for i, j in self.active_aircraft_combinations:
            if i != j:
                self.collisions[i, j] = self.collisions[j, i] = self.aircraft[i] & self.aircraft[j]

        self.conflicts[:, :] = conflicts_between_multiple(self, t_lookahead=self.t_lookahead)

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


class CombinedFlows:
    t_lookahead = 5/60

    def __init__(self, flows: OrderedDict):
        self.flows = copy.deepcopy(flows)
        self.flow_keys = self.flows.keys()
        self.flow_key_pairs = list(itertools.combinations(self.flow_keys, 2))
        self.current_conflict_state = {
            (flow_a, flow_b): np.zeros((self.flows[flow_a].n, self.flows[flow_b].n), dtype=bool)
            for flow_a, flow_b in self.flow_key_pairs}
        self.current_conflict_state.update({k: self.flows[k].conflicts for k in self.flow_keys})
        self._update_conflicts()

    def _update_conflicts(self):
        for flow_a, flow_b in self.flow_key_pairs:
            conflicts_between_multiple(self.flows[flow_a], self.flows[flow_b], t_lookahead=self.t_lookahead,
                                       out=self.current_conflict_state[(flow_a, flow_b)])
        # Inter-flow conflicts should be updated by the respective flows _update_collisions_and_conflicts

    def step(self, dt):
        dt = Aircraft.convert_dt(dt)
        for k in self.flow_keys:
            self.flows[k].step(dt)
        self._update_conflicts()


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


def conflicts_between_multiple(own: Flow, other: Flow=None, t_lookahead=5. / 60, out=None):
    if other is None:
        other = own
    x_rel = -(np.expand_dims(own.position, 1) - np.expand_dims(other.position, 0))
    v_rel = np.expand_dims(own.v, 1) - np.expand_dims(other.v, 0)
    both_active = np.expand_dims(own.active, 1) & np.expand_dims(other.active, 0)
    x_v_inner = np.einsum('ijk,ijk->ij', x_rel, v_rel)
    v_rel_norm_sq = norm(v_rel, axis=2)**2
    x_rel_norm_sq = norm(x_rel, axis=2)**2
    with np.errstate(invalid='ignore'):
        t_horizontal_conflict = np.array([(x_v_inner + np.sqrt(x_v_inner**2 - x_rel_norm_sq*v_rel_norm_sq+v_rel_norm_sq * Aircraft.horizontal_separation_requirement**2))/v_rel_norm_sq,
                                          (x_v_inner - np.sqrt(x_v_inner ** 2 - x_rel_norm_sq * v_rel_norm_sq + v_rel_norm_sq * Aircraft.horizontal_separation_requirement ** 2)) / v_rel_norm_sq])
        t_cpa = np.average(t_horizontal_conflict, axis=0)
        minimum_distance = norm(np.einsum('ij,ijk->ijk', t_cpa, v_rel) - x_rel, axis=2)
        t_in_hor = np.min(t_horizontal_conflict, axis=0)
        t_out_hor = np.max(t_horizontal_conflict, axis=0)
        horizontal_conflict = (minimum_distance < Aircraft.horizontal_separation_requirement) \
                              & (t_out_hor > 0)\
                              & (t_in_hor < t_lookahead)
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
    print(conflicts_between_multiple(flow1, flow2, t_lookahead=7/20-0.01))
