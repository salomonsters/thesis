import itertools
import logging

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

    def step(self, dt):
        if isinstance(dt, Quantity):
            dt = dt.to(ureg.hour).magnitude
        dt = self.dtype(dt)
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
    def __init__(self, index, position, trk, gs, alt, vs, callsign, active:np.ndarray):
        self.index = index
        super().__init__(position[index:index + 1], trk[index:index + 1], gs[index:index + 1], alt[index:index + 1],
                         vs[index:index + 1], callsign[index:index + 1], active[index:index + 1])

    def step(self, *args, **kwargs):
        raise RuntimeError("Step is not implemented for AircraftInFlow")


class Flow:
    callsign_map = None

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

        self._calculate_active_aircraft_combinations()
        self._update_collisions()

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
        self._update_collisions()

    def deactivate(self, callsign):
        return self.activate(callsign, True)

    def step(self, dt):
        self.position[self.active] += self.v[self.active] * dt
        self.alt[self.active] += self.vs_fph[self.active] * dt
        self._update_collisions()

    def _update_collisions(self):
        # Inactive aircraft have no collisions
        self.collisions[:, :] = False
        for i, j in self.active_aircraft_combinations:
            if i != j:
                self.collisions[i, j] = self.collisions[j, i] = self.aircraft[i] & self.aircraft[j]

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


def conflict_between(own:Aircraft, intruder:Aircraft, t_lookahead=5./60):
    x_rel = (intruder.position - own.position).reshape((2,))
    v_rel = (own.v - intruder.v).reshape((2,))

    t_horizontal_conflict = (np.inner(x_rel, v_rel) + np.array((-1, 1)) * np.sqrt(np.inner(x_rel, v_rel)**2 - norm(x_rel)**2*norm(v_rel)**2+norm(v_rel)**2*Aircraft.horizontal_separation_requirement**2))/(norm(v_rel)**2)
    t_cpa = np.sum(t_horizontal_conflict)/2.
    mindist = norm(t_cpa*v_rel - x_rel)
    t_in_hor, t_out_hor = min(t_horizontal_conflict), max(t_horizontal_conflict)
    horizontal_conflict = mindist < Aircraft.horizontal_separation_requirement and t_out_hor > 0 and t_in_hor < t_lookahead
    if not horizontal_conflict:
        return False

    vs_rel_fph = own.vs_fph - intruder.vs_fph
    alt_diff = intruder.alt - own.alt

    if np.abs(vs_rel_fph) < 1e-8:
        if np.abs(alt_diff) < Aircraft.vertical_separation_requirement:
            # We have a horizontal conflict and we are flying level at conflicting altitudes, so we have a conflict
            return True

    t_vertical_conflict = (intruder.alt - own.alt + np.array((-1, 1))*Aircraft.vertical_separation_requirement)/vs_rel_fph
    t_in_vert, t_out_vert = min(t_vertical_conflict), max(t_vertical_conflict)

    t_in_combined = max(t_in_hor, t_in_vert)
    # t_out_combined = max(t_out_hor, t_out_vert)

    if t_in_combined < t_lookahead:
        return True
    return False


if __name__ == "__main__":
    pass
