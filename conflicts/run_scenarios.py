import contextlib
from collections import OrderedDict

import numpy as np

from conflicts.simulate import generate_start_positions, Aircraft, Flow, run_and_save_sim_results_to_file
from conflicts.visualize_simulation import plot_simulation_consistency
import conflicts.simulate


radius = 200
gs = 200
n_aircraft_per_flow = 100
calculate_conflict_rate = True
T_conflict_window = [2.5, 5]
f_plot = None
f_sim = 3600
f_conflict = f_sim // 240
acph = 15
lam = acph / f_sim
n_realisations = 10
filename_suffix = "-Tc_2.5_5-"


def lambda_experiment(save_plot=False, do_calculations=True):
    n_aircraft_per_flow = 200

    filenames = []
    for realisation in range(n_realisations):

        out_fn = 'data/simulated_conflicts/' \
                 'poisson-f-3600-gs-{}_trk-0-30-120_vs-0-acph-3-2-18-R_{}nm-realisation-{}.xlsx'.format(gs, radius, realisation)
        filenames.append(out_fn)
        if do_calculations:
            flows_kwargs = OrderedDict()
            flows_dict = OrderedDict()

            flow_i = 0

            for trk in (0, 30, 60, 90, 120):
                for lam in list(np.arange(3, 18, 2) / f_sim):
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
                            # 'measured_distances_at_spawn': np.zeros((n_aircraft_per_flow,), dtype=float),
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
        plot_fn = 'pgf/varying_isolated_quantities/lambda-{}-realisations_gs_{}-R_{}nm.{}'.format(n_realisations, gs, radius, save_plot)
        plot_options = {
            'fn': plot_fn,
            'size': {'w': 3.5, 'h': 6}
        }

        plot_simulation_consistency(filenames, plot_options=plot_options)


def gs_experiment(save_plot=False, do_calculations=True):
    filenames = []
    for realisation in range(n_realisations):

        out_fn = 'data/simulated_conflicts/' \
                 'poisson-f-3600-gs-100-20-400_trk-0-30-120_vs-0-acph-{}-R_{}nm-realisation-{}.xlsx'.format(acph, radius, realisation)
        filenames.append(out_fn)
        if do_calculations:
            flows_kwargs = OrderedDict()
            flows_dict = OrderedDict()

            flow_i = 0

            for trk in (0, 30, 60, 90, 120):
                for gs in list(np.arange(100, 400, 20)):
                    flow_name = 'trk_{}_gs_{}'.format(int(trk), int(gs))

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
                            # 'measured_distances_at_spawn': np.zeros((n_aircraft_per_flow,), dtype=float),
                            'IV': 'gs',
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
        plot_fn = 'pgf/varying_isolated_quantities/vary_gs_100_20_400-{}_realisations-{}_acph-R_{}nm.{}'.format(n_realisations, acph, radius, save_plot)
        plot_options = {
            'fn': plot_fn,
            'size': {'w': 3.5, 'h': 6}
        }

        plot_simulation_consistency(filenames, plot_options=plot_options)


def gs_experiment_lower_V(save_plot=False, do_calculations=True):
    filenames = []
    for realisation in range(n_realisations):

        out_fn = 'data/simulated_conflicts/' \
                 'poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0-acph-{}-R_{}nm-realisation-{}{}.xlsx'.format(acph, radius, realisation, filename_suffix)
        filenames.append(out_fn)
        if do_calculations:
            flows_kwargs = OrderedDict()
            flows_dict = OrderedDict()

            flow_i = 0

            for trk in (0, 30, 60, 90, 120):
                for gs in list(np.arange(100, 201, 5)):
                    flow_name = 'trk_{}_gs_{}'.format(int(trk), int(gs))

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
                            # 'measured_distances_at_spawn': np.zeros((n_aircraft_per_flow,), dtype=float),
                            'IV': 'gs',
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
        plot_fn = 'pgf/varying_isolated_quantities/vary_gs_100_5_200-{}_realisations-{}_acph-R_{}nm{}.{}'.format(n_realisations, acph, radius, filename_suffix, save_plot)
        plot_options = {
            'fn': plot_fn,
            'size': {'w': 3.5, 'h': 6}
        }

        plot_simulation_consistency(filenames, plot_options=plot_options)



def separation_requirement_experiment(save_plot=False, do_calculations=True):
    S_h = 5  # nm
    global conflicts
    original_S_h = conflicts.simulate.Aircraft.horizontal_separation_requirement
    conflicts.simulate.Aircraft.horizontal_separation_requirement = S_h
    filenames = []
    for realisation in range(n_realisations):

        out_fn = 'data/simulated_conflicts/' \
                 'poisson-f-3600-gs-100-20-400_trk-0-30-120_vs-0-acph-{}-R_{}nm-Sh_{}-realisation-{}.xlsx'.format(acph, radius, S_h, realisation)
        filenames.append(out_fn)
        if do_calculations:
            flows_kwargs = OrderedDict()
            flows_dict = OrderedDict()

            flow_i = 0

            for trk in (0, 30, 60, 90, 120):
                for gs in list(np.arange(100, 400, 20)):
                    flow_name = 'trk_{}_gs_{}'.format(int(trk), int(gs))

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
                            # 'measured_distances_at_spawn': np.zeros((n_aircraft_per_flow,), dtype=float),
                            'IV': 'gs',
                            'S_h': S_h,
                        }
                    }
                    flows_dict[flow_name] = Flow.expand_properties(flows_kwargs[flow_name])
                    flow_i += 1
            with contextlib.redirect_stdout(None):
                run_and_save_sim_results_to_file(flows_kwargs, flows_dict, out_fn, T_conflict_window, f_simulation=f_sim,
                                                 f_plot=None, stop_function=None, try_save_on_exception=False,
                                                 calculate_conflict_per_time_unit=calculate_conflict_rate)
            print("Saved realisation {} to {}".format(realisation, out_fn))
    # Reset S_h
    conflicts.simulate.Aircraft.horizontal_separation_requirement = original_S_h
    if save_plot:
        plot_fn = 'pgf/varying_isolated_quantities/vary_gs_100_20_400-{}_realisations-{}_acph-R_{}nm-Sh_{}.{}'.format(n_realisations, acph, radius, S_h, save_plot)
        plot_options = {
            'fn': plot_fn,
            'size': {'w': 3.5, 'h': 6}
        }

        return plot_simulation_consistency(filenames, plot_options=plot_options)


def flow_intersection_experiment(save_plot=False, do_calculations=True):
    # radius = 200
    filenames = []
    for realisation in range(n_realisations):

        out_fn = 'data/simulated_conflicts/' \
                 'poisson-f-3600-gs-{}_trk-0-2.5-360_vs-0-acph-{}-R_{}nm-realisation-{}.xlsx'.format(gs, acph, radius, realisation)
        filenames.append(out_fn)
        if do_calculations:
            flows_kwargs = OrderedDict()
            flows_dict = OrderedDict()

            flow_i = 0

            for trk in list(np.arange(0, 360, 2.5)):
                flow_name = 'trk_{}'.format(int(trk))

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
                        # 'measured_distances_at_spawn': np.zeros((n_aircraft_per_flow,), dtype=float),
                        'IV': 'trk_diff',
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
        plot_fn = 'pgf/varying_isolated_quantities/vary_trk_0_2.5_360-{}_realisations-gs_{}-{}_acph-R_{}nm.{}'.format(n_realisations, gs, acph, radius, save_plot)
        plot_options = {
            'fn': plot_fn,
            'size': {'w': 3.5, 'h': 6}
        }

        return plot_simulation_consistency(filenames, plot_options=plot_options)


if __name__ == "__main__":
    from tools import create_logger

    log = create_logger(verbose=True)
    kwargs = {"save_plot": "eps", "do_calculations": True}
    jobs = [
        # lambda_experiment,
        #gs_experiment,
        gs_experiment_lower_V,
        # flow_intersection_experiment,
        # separation_requirement_experiment
    ]
    for job in jobs:
        log("Starting {}".format(job))
        try:
            result = job(**kwargs)
        except:
            log("Exception occured, continuing with the next experiment")

