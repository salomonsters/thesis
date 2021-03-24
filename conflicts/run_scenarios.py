from collections import OrderedDict
import contextlib

import numpy as np

from conflicts.simulate import generate_start_positions, Aircraft, Flow, run_and_save_sim_results_to_file
from conflicts.visualize_simulation import plot_simulation_consistency

radius = 20
gs = 100
n_aircraft_per_flow = 200
calculate_conflict_rate = True
T_conflict_window = [1, 5]
f_plot = None
f_sim = 3600
f_conflict = f_sim // 240

def lambda_experiment(save_plot=False):

    filenames = []
    for realisation in range(10):

        out_fn = 'data/simulated_conflicts/' \
                 'poisson-f-3600-{}-200_trk-0-30-120_vs-0-acph-3-2-18-realisation-{}.xlsx'.format(gs, realisation)
        filenames.append(filenames)
        flows_kwargs = OrderedDict()
        flows_dict = OrderedDict()

        flow_i = 0

        for trk in (0, 30, 60, 90, 120):
            for lam in list(np.arange(3, 18, 2)/f_sim):
                flow_name = 'trk_{}_lam_{:.4f}'.format(int(trk), lam)

                x0, y0 = generate_start_positions((0, 0), radius, trk)

                flows_kwargs[flow_name] = {
                    'position': (x0, y0),
                    'trk': trk,
                    'gs': gs,
                    'alt': 2000,
                    'vs': 0,
                    'callsign': ['flow_{0}_ac_{1}'.format(trk, i) for i in range(n_aircraft_per_flow)],
                    'active': False,
                    'other_properties': {
                        'lam': lam,
                        'conflict_rate_calculated': calculate_conflict_rate,
                        'measured_distances_at_spawn': np.zeros((n_aircraft_per_flow, ), dtype=float),
                        'IV': 'lam',
                        'S_h': Aircraft.horizontal_separation_requirement,
                    }
                }
                flows_dict[flow_name] = Flow.expand_properties(flows_kwargs[flow_name])
                flow_i += 1
        with contextlib.redirect_stdout(None):
            run_and_save_sim_results_to_file(flows_kwargs, flows_dict, out_fn, T_conflict_window, f_simulation=f_sim,
                                         f_plot=None, stop_function=None, try_save_on_exception=False,
                                         calculate_conflict_per_time_unit=calculate_conflict_rate)
        print("Saved realisation {} to {}".format(realisation, out_fn))
    if save_plot:
        plot_fn = 'pgf/varying_isolated_quantities/lambda-10-realisations_gs_{}.pgf'.format(gs)
        pgf_options = {
            'fn': plot_fn,
            'size': {'w': 3.5, 'h': 6}
        }

        plot_simulation_consistency(filenames, pgf_options=pgf_options)


if __name__ == "__main__":
    from tools import create_logger
    log = create_logger(verbose=True)
    kwargs = {"save_plot": True}
    jobs = [
        lambda_experiment
    ]
    for job in jobs:
        log("Starting {}".format(job))
        job(**kwargs)
