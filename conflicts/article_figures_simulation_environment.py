import copy
from collections import OrderedDict

import matplotlib.pyplot as plt

import numpy as np

from conflicts.simulate import Aircraft, Flow, AircraftInFlow, CombinedFlows, Simulation

class SimulationForArticle(Simulation):
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
            for i in range(self.flows[flow].active.shape[0]):
                if self.flows[flow].active[i]:
                    annotation = flow
                    plt.annotate(annotation,
                                 (self.flows[flow].position[i, 0], self.flows[flow].position[i, 1]),
                                 # this is the point to label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(0, 10),  # distance from text to points (x,y)
                                 ha='left')  # horizontal alignment can be left, right or center

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        # plt.legend()
        if self.t > 0:
            # progress = self.t / self.T
            # progress_divider = 1/self.lw_conflict
            # title = int(progress // progress_divider) * '#' +int((1 - progress) // progress_divider )* '_'
            title = f"{self.t=:.4f}"

            if self.savefig_str:
                plt.axis('off')
                plt.gcf().set_size_inches((3.5, 3.5))
                plt.savefig(self.savefig_str.format(self.t), bbox_inches='tight')
            else:
                plt.title(title)
        plt.pause(0.05)



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
sim = SimulationForArticle(combined_flows, plot_frequency=20,
                 savefig_str='pgf/simulation_environment_example-{:.4f}.pgf')
sim.simulate(20, 0.7, T_conflict_window=[0, 0.2])

from conflicts.run_scenarios import gs, lam, f_sim
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib

trk_deg = np.arange(0, 180, 2.5)
trk_rad = np.radians(trk_deg)
B_exp = gs / (lam * f_sim)
Vrh = (gs ** 2 + gs ** 2 - 2 * gs * gs * np.cos(trk_rad)) ** 0.5
pred_conflict_rate = 2 * Aircraft.horizontal_separation_requirement / (B_exp * B_exp) * (Vrh / (np.sin(trk_rad)))

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})
pgf_options = {
    'fn': 'pgf/predicted_conflicts_vs_trk_diff.pgf',
    'size': {'w': 3.5, 'h': 2.5}
}
fig, ax = plt.subplots(1,1, figsize=(6.5, 6.5))
plt.plot(pred_conflict_rate, trk_deg)
ax.set_xlabel("Predicted conflict rate [1/hr]")
ax.set_ylabel('Angle between flows [$^\circ$]')
ax.yaxis.set_major_locator(ticker.MultipleLocator(20))
fig.set_size_inches(**pgf_options['size'])
fig.savefig(pgf_options['fn'], bbox_inches='tight')