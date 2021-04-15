import ast
import os
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import conflicts.simulate
from conflicts.simulate import Flow, Simulation, CombinedFlows

rng = np.random.default_rng()


class ReplayFlow(Flow):
    usable_columns = ('ts', 'alt', 'gs', 'trk', 'x', 'y', 'roc', 'fid')

    def __init__(self, df, callsign_col, cluster_name, delete_after=110, time_shift_max=None, activate_based_on_flow_lambda=False):
        groupby_callsign = df.groupby(callsign_col)
        callsigns = df.groupby(callsign_col).size().index

        n = len(callsigns)
        position = np.zeros((n, 2))
        trk = np.zeros(n)
        gs = np.zeros(n)
        alt = np.zeros(n)
        vs = np.zeros(n)
        active = np.zeros(n, dtype=bool)
        active[:] = False
        other_properties = {'cluster_name': cluster_name}
        self.cluster_name = cluster_name
        self.df = df.loc[:, self.usable_columns]
        self.delete_after = delete_after
        self.time_in_seconds = 0

        self.t_activate = np.zeros(n)
        self.t_deactivate = np.zeros(n)
        self.dataframes = []

        assert not (time_shift_max is not None and activate_based_on_flow_lambda), \
            "invalid timeshift/activation parameters"

        self.time_shifts = None

        if time_shift_max is not None:
            self.time_shifts = rng.uniform(0, time_shift_max, n).astype(int)

        elif activate_based_on_flow_lambda:
            df_ts_0 = self.df.groupby(callsign_col)['ts'].min()
            mean_time_between_activations = df_ts_0.sort_values().diff()[1:].mean()
            self.activation_times = np.cumsum(rng.exponential(scale=mean_time_between_activations, size=n))
            rng.shuffle(self.activation_times)

            new_ts_0 = pd.Series(self.activation_times, index=callsigns)
            self.time_shifts = (new_ts_0 - df_ts_0).to_numpy()

        i = 0
        for callsign, df_callsign in groupby_callsign:
            assert callsigns[i] == callsign
            if self.time_shifts is not None:
                df_callsign['ts'] += self.time_shifts[i]
                self.df.loc[df_callsign.index, 'ts'] = df_callsign['ts']
                # assert np.abs(df_callsign['ts'].min() - self.activation_times[i])<1

            self.t_activate[i] = df_callsign['ts'].min()
            self.t_deactivate[i] = df_callsign['ts'].max() + delete_after
            self.dataframes.append(df_callsign)
            i += 1


        super().__init__(position, trk, gs, alt, vs, callsigns, active, calculate_collisions=False,
                         other_properties=other_properties)

        self.already_deactivated = np.zeros_like(self.active)
        self.already_deactivated[:] = False
        self.callsigns = callsigns

    @classmethod
    def expand_properties(cls, *args, **kwargs):
        raise NotImplementedError()

    def step(self, dt, update_conflicts=True, t=None):
        # raise NotImplementedError("Moet nog gedaan worden")
        # dt = Aircraft.convert_dt(dt) # this gives a dt in hours
        # dt_seconds = dt/3600.
        # self.position[self.active] += self.v[self.active] * dt
        # self.alt[self.active] += self.vs_fph[self.active] * dt
        t_in_seconds = t * 3600
        for i in self.index:
            if not self.already_deactivated[i] and not self.active[i]:
                if t_in_seconds >= self.t_activate[i]:
                    self.activate(self.callsign[i])
                    print("{}h (simtime): Activating flow {} ac {}".format(t, self.cluster_name, self.callsign[i]))
            elif self.active[i]:
                if t_in_seconds >= self.t_deactivate[i]:
                    self.deactivate(self.callsign[i])
                    self.already_deactivated[i] = True
                    print("{}h (simtime): Deactivating flow {} ac {}".format(t, self.cluster_name, self.callsign[i]))
                else:
                    idx = self.dataframes[i]['ts'].sub(t_in_seconds).abs().idxmin()
                    row = self.dataframes[i].loc[idx]
                    # print(row)
                    self.position[i] = row[['x', 'y']] / 1852  # Convert position coordinates to nautical miles
                    self.trk[i] = row['trk']
                    self.gs[i] = row['gs']
                    self.alt[i] = row['alt']
                    self.vs[i] = row['roc']

                    self.aircraft[i].calculate_speed_vector()
        self.v = (self.gs * np.array([np.sin(np.radians(self.trk)), np.cos(np.radians(self.trk))])).T
        self.vs_fph = self.vs * 60.

        if update_conflicts:
            self._update_collisions_and_conflicts()


class ReplaySimulation(Simulation):
    def __init__(self, *args, **kwargs):
        self.replay_df = kwargs.pop('replay_df')
        super().__init__(*args, **kwargs)

    def prepare_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        plt.ion()
        self.xlim = np.array([self.replay_df['x'].min(), self.replay_df['x'].max()]) / 1852
        self.ylim = np.array([self.replay_df['y'].min(), self.replay_df['y'].max()]) / 1852
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
        any_active_flows = False
        for flow_i, flow in enumerate(self.flows.flow_keys):
            active_conflicts = self.flows.active_conflicts_within_flow_or_between_flows[flow][self.flows[flow].active]
            if active_conflicts.shape[0] == 0:
                continue
            any_active_flows = True
            plt.plot(self.flows[flow]['position'][active_conflicts][:, 0],
                     self.flows[flow]['position'][active_conflicts][:, 1], lw=0, marker='o', fillstyle='none',
                     c='C{}'.format(flow_i), markersize=self.lw_conflict)
            plt.plot(self.flows[flow]['position'][~active_conflicts][:, 0],
                     self.flows[flow]['position'][~active_conflicts][:, 1], lw=0, marker='o', fillstyle='none',
                     c='C{}'.format(flow_i), markersize=self.lw_no_conflict, label=flow)
            for i in range(self.flows[flow].active.shape[0]):
                if self.flows[flow].active[i]:
                    plt.annotate("{:.0f}ft".format(self.flows[flow].alt[i]),  # this is the text
                                 (self.flows[flow].position[i, 0], self.flows[flow].position[i, 1]),
                                 # this is the point to label
                                 textcoords="offset points",  # how to position the text
                                 xytext=(0, 10),  # distance from text to points (x,y)
                                 ha='left')  # horizontal alignment can be left, right or center

            plt.quiver(self.flows[flow]['position'][:, 0], self.flows[flow]['position'][:, 1],
                       self.flows[flow]['v'][:, 0], self.flows[flow]['v'][:, 1], color='C{}'.format(flow_i),
                       angles='xy', scale_units='xy', scale=1 / Flow.t_lookahead, width=0.001)

        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        if any_active_flows:
            plt.legend()
        if self.t > 0:
            # progress = self.t / self.T
            # progress_divider = 1/self.lw_conflict
            # title = int(progress // progress_divider) * '#' +int((1 - progress) // progress_divider )* '_'
            title = f"{self.t=}"
            plt.title(title)
        plt.pause(0.05)


if __name__ == "__main__":
    t_lookahead = 10./60
    conflicts.simulate.S_h = 3
    conflicts.simulate.S_v = 1000
    f_simulation = 3600 // 6
    f_plot = None#3600 // 60
    f_conflict = 3600 // 6
    do_calc = True
    do_plot = False
    do_save = True
    T = 24  # hrs
    n_splits = 4
    disregard_days = True
    as_events = True
    copy_events = None # times
    start_midday = False
    data_date_fmt = '20180101-20180102-20180104-20180105_split_{}'
    filename_prefix = 'eham_stop_mean_0.25_std_0.25'
    # max_time_shift = 3600 # seconds
    splits_to_consider = list(range(n_splits))

    events_suffix = ""
    if as_events:
        events_suffix = "-as-events"
        if copy_events:
            events_suffix = "-as-events-repeats-{}".format(copy_events)

    from tools import create_logger
    log = create_logger(verbose=True)
    if do_plot:
        heatmap_list = []
        fig, ax = plt.subplots(2, 2)
        ax = ax.reshape((n_splits,))
    records = []
    # results_fn = 'data/conflict_replay_results/eham_stop_mean_std_0.28_20180101-20180102-20180104-20180105-splits_{}-S_h-{}-S_v-{}-t_l-{:.4f}-as-events-repeats-{}.xlsx'.format(repr(splits_to_consider).replace(', ','-'), conflicts.simulate.S_h, conflicts.simulate.S_v, t_lookahead, copy_events)
    results_fn = 'data/conflict_replay_results/{}_20180101-20180102-20180104-20180105-splits_{}-S_h-{}-S_v-{}-t_l-{:.4f}{}.xlsx'.format(filename_prefix, repr(splits_to_consider).replace(', ','-'), conflicts.simulate.S_h, conflicts.simulate.S_v, t_lookahead, events_suffix)
    for split in splits_to_consider:

        log("Starting split {}".format(split))
        data_date = data_date_fmt.format(split)
        callsign_col = 'fid'

        # out_fn = 'data/replay_conflicts/eham_stop_mean_std_0.28_{}-fsim-{}-fconflict-{}-S_h-{}-S_v-{}-t_l-{:.4f}-as-events-repeats-{}.xlsx'.format(data_date, f_simulation, f_conflict, conflicts.simulate.S_h, conflicts.simulate.S_v, t_lookahead, copy_events)
        out_fn = 'data/replay_conflicts/{}_{}-fsim-{}-fconflict-{}-S_h-{}-S_v-{}-t_l-{:.4f}{}.xlsx'.format(filename_prefix, data_date, f_simulation, f_conflict, conflicts.simulate.S_h, conflicts.simulate.S_v, t_lookahead, events_suffix)

        if do_calc:
            save_events_repeat_file = False

            events_repeat_filename = 'data/clustered/{}_{}{}.csv'.format(filename_prefix, data_date, events_suffix)
            if copy_events and os.path.exists(events_repeat_filename):
                df = pd.read_csv(events_repeat_filename)
            else:
                if copy_events:
                    save_events_repeat_file = True
                with open('data/clustered/{}_{}.csv'.format(filename_prefix, data_date), 'r') as fp:
                    parameters = ast.literal_eval(fp.readline())
                    df = pd.read_csv(fp)
                # df = df[~pd.isna(df['trk'])]
                ts_0 = df['ts'].min()
                if copy_events:
                    # df_orig = df.copy(deep=True)
                    # df[callsign_col] = df[callsign_col] + '_0'

                    df_list = []
                    for fid, fid_df in df.groupby(callsign_col):
                        for repeat_i in range(copy_events):
                            df_list.append(fid_df.assign(**{callsign_col: '{}_{}'.format(fid, repeat_i)}))
                    df = pd.concat(df_list)
                    df = df.reset_index(drop=True)#.sort_values(by=[callsign_col, 'ts'])
                if start_midday:
                    df.query('ts-{}>12*3600'.format(df['ts'].min()), inplace=True)

                df['ts'] = df['ts'] - ts_0

                df['t'] = pd.TimedeltaIndex(df['ts'], unit='s')
                if disregard_days:
                    df['t'] = df['t'] - df['t'].dt.floor('D')
                    df['ts'] = df['t'].dt.total_seconds()

                df.sort_values(by=['cluster', callsign_col, 't'], inplace=True)
            flows_dict = {}
            for cluster, group in df.groupby('cluster'):

                flows_dict[cluster] = ReplayFlow(group, callsign_col, cluster, delete_after=50, activate_based_on_flow_lambda=as_events)
                flows_dict[cluster].t_lookahead = t_lookahead
                if flows_dict[cluster].time_shifts is not None:
                    df.loc[flows_dict[cluster].df.index, 'ts'] = flows_dict[cluster].df['ts']
            flows = CombinedFlows(OrderedDict(flows_dict))
            flows.t_lookahead = t_lookahead
            if copy_events and save_events_repeat_file:
                log("Converted df to {} repeats".format(copy_events))
                df.to_csv(events_repeat_filename, index=False)
                log('Saved repeated df to {}'.format(events_repeat_filename))


            sim = ReplaySimulation(flows, plot_frequency=f_plot, calculate_conflict_per_time_unit=False, replay_df=df)
            sim.simulate(f_simulation, conflict_frequency=f_conflict, T=T)

            import pandas as pd

            df_conflicts = pd.DataFrame.from_records([(v, *k) if isinstance(k, tuple) else (v, k) for k, v
                                                      in sim.aggregated_conflicts.items()],
                                                     columns=('conflicts', 'flow1', 'flow2'))[
                ['flow1', 'flow2', 'conflicts']]
            with pd.ExcelWriter(out_fn) as writer:

                df_conflicts.to_excel(writer, sheet_name='Conflicts')
                # df2.to_excel(writer, sheet_name='Properties')
            print("Results saved to {}".format(out_fn))

        else:
            df_conflicts = pd.read_excel(out_fn)
        n_cluster = df_conflicts['flow1'].max() + 1  # +1 for unclustered

        between_flow_conflicts = np.zeros((n_cluster, n_cluster))

        # Loop through rows, change -1 (unclustered) to 0. Assert 0 doesn't exist as cluster number for i or j.
        for _, row in df_conflicts.iterrows():
            i = int(row['flow1'])
            assert i != 0
            if i == -1:
                i = 0
            try:
                j = int(row['flow2'])
                assert j != 0
                if j == -1:
                    j = 0
                between_flow_conflicts[i, j] = row['conflicts']
                records.append({'i': i, 'j': j, 'type': 'between', 'unclustered': i == 0, 'split': split, 'conflicts': row['conflicts']})
            except ValueError:
                # in this case j=int(np.nan) so we get a valueerror
                # (ab)use this to set within-flow conflicts
                between_flow_conflicts[i, i] = row['conflicts']
                records.append({'i': i, 'j': np.nan, 'type': 'within', 'unclustered': i == 0, 'split': split,
                                'conflicts': row['conflicts']})
        if do_plot:
            mask = np.ones_like(between_flow_conflicts)

            mask[between_flow_conflicts > 0] = False
            mask[0,:] = True
            sns.heatmap(between_flow_conflicts, mask=mask, cmap="flare", ax=ax[split])

            heatmap_list.append(between_flow_conflicts)
    df_results = pd.DataFrame(records)
    if do_plot:
        plt.show()

        plt.figure()
        df_results.boxplot(column='conflicts', by=['type', 'unclustered'])
        plt.show()
        plt.figure()
        df_results.query('unclustered == False').boxplot(column='conflicts', by=['split', 'type'])
        plt.show()
    if do_save:
        df_results.to_excel(results_fn)
