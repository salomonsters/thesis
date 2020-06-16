import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

class Aircraft:
    horizontal_separation_requirement = 3       # nm
    vertical_separation_requirement = 1000      # ft
    dtype = np.float

    def __init__(self, position, trk, gs, alt, vs, callsign, active=True):
        self.position = np.array(position, dtype=self.dtype)   # nm
        self._gs = gs                                           # knots
        self.trk = trk                                          # degrees
        self.alt = alt                                          # feet
        self.vs = vs                                            # feet per minute
        self.callsign = callsign
        self.active = active

    def _update_speed_or_track(self):
        self.vx = self._gs * np.sin(self._trk_rad)              # knots
        self.vy = self._gs * np.cos(self._trk_rad)              # knots
        self.v = np.array([self.vx, self.vy])                   # knots

    @property
    def gs(self):
        return self._gs

    @gs.setter
    def gs(self, gs):
        self._gs = gs
        self._update_speed_or_track()

    @property
    def trk(self):
        return self._trk

    @trk.setter
    def trk(self, trk):
        self._trk = trk
        self._trk_rad = np.radians(trk)
        self._update_speed_or_track()

    @property
    def vs(self):
        return self._vs_fpm

    @vs.setter
    def vs(self, vs):
        self._vs_fpm = vs
        self.vs_fph = vs * 60.

    def step(self, dt):
        dt = self.dtype(dt)
        if self.active:
            self.position += self.v * dt
            self.alt += self.vs_fph * dt

    def __and__(self, other):
        if not isinstance(other, type(self)):
            raise NotImplementedError()
        return np.linalg.norm(self.position - other.position) <= self.horizontal_separation_requirement and\
               np.abs(self.alt - other.alt) <= self.vertical_separation_requirement


class Flow:
    def __init__(self, aircraft_list):
        if not np.all(map(lambda ac: isinstance(ac, Aircraft), aircraft_list)):
            raise ValueError()
        self.callsign_map = {aircraft.callsign: i for i, aircraft in enumerate(aircraft_list)}
        dtype = Aircraft.dtype
        self.position = np.array([aircraft.position for aircraft in aircraft_list], dtype=dtype)
        self.v = np.array([aircraft.v for aircraft in aircraft_list], dtype=dtype)
        self.alt = np.array([aircraft.alt for aircraft in aircraft_list], dtype=dtype)
        self.vs_fph = np.array([aircraft.vs_fph for aircraft in aircraft_list], dtype=dtype)
        self.active = np.array([aircraft.active for aircraft in aircraft_list], dtype=bool)

    def activate(self, callsign, deactivate=False):
        if deactivate:
            if not self.active[self.callsign_map[callsign]]:
                logging.warning("Aircraft {} was already inactive".format(callsign))
            self.active[self.callsign_map[callsign]] = False
        else:
            if self.active[self.callsign_map[callsign]]:
                logging.warning("Aircraft {} was already active".format(callsign))
            self.active[self.callsign_map[callsign]] = True

    def deactivate(self, callsign):
        return self.activate(callsign, True)

    def step(self, dt):
        self.position[self.active] += self.v[self.active] * dt
        self.alt[self.active] += self.vs_fph[self.active] * dt


if __name__ == "__main__":
    pass
