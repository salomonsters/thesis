import ast
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from conflicts.simulate import Flow, Simulation, CombinedFlows


class ReplayFlow(Flow):
    usable_columns = ('ts', 'alt', 'gs', 'trk', 'x', 'y', 'roc')

    def __init__(self, df, callsign_col, cluster_name, delete_after=110):
        callsigns = df[callsign_col].unique()

        n = callsigns.shape[0]
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
        super().__init__(position, trk, gs, alt, vs, callsigns, active, calculate_collisions=False,
                         other_properties=other_properties)
        self.t_activate = np.zeros(n)
        self.t_deactivate = np.zeros(n)
        self.dataframes = []
        for i, callsign in enumerate(callsigns):
            df_callsign = df[df[callsign_col] == callsign]
            self.t_activate[i] = df_callsign['ts'].min()
            self.t_deactivate[i] = df_callsign['ts'].max() + delete_after
            self.dataframes.append(df_callsign)
        self.already_deactivated = np.zeros_like(self.active)
        self.already_deactivated[:] = False

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
    f_simulation = 3600 // 6
    f_plot = None#3600 // 60
    f_conflict = 3600 // 6
    do_calc = False
    T = 24  # hrs
    n_splits = 4
    disregard_days = True
    start_midday = False
    data_date_fmt = '20180101-20180102-20180104-20180105_split_{}'
    results_fn = 'data/conflict_replay_results/20180101-20180102-20180104-20180105-splits-1-to-4.xlsx'


    from tools import create_logger
    log = create_logger(verbose=True)
    heatmap_list = []
    fig, ax = plt.subplots(2, 2)
    ax = ax.reshape((n_splits,))
    records = []
    for split in list(range(n_splits)):
        log("Starting split {}".format(split))
        data_date = data_date_fmt.format(split)
        callsign_col = 'fid'

        out_fn = 'data/replay_conflicts/eham_{}-fsim-{}-fconflict-{}.xlsx'.format(data_date, f_simulation, f_conflict)

        if do_calc:
            with open('data/clustered/eham_{0}.csv'.format(data_date), 'r') as fp:
                parameters = ast.literal_eval(fp.readline())
                df = pd.read_csv(fp)
            df = df[~pd.isna(df['trk'])]
            if start_midday:
                df.query('ts-{}>12*3600'.format(df['ts'].min()), inplace=True)
            ts_0 = df['ts'].min()
            df['ts'] = df['ts'] - ts_0

            df['t'] = pd.TimedeltaIndex(df['ts'], unit='s')
            if disregard_days:
                df['t'] = df['t'] - df['t'].dt.floor('D')
                df['ts'] = df['t'].dt.total_seconds()

            df.sort_values(by=['cluster', callsign_col, 't'], inplace=True)
            flows_dict = {}
            for cluster, group in df.groupby('cluster'):
                flows_dict[cluster] = ReplayFlow(group, callsign_col, cluster, delete_after=50)
                # break
                # for _ in range(60*24):
                #     flows_dict[cluster].step(1/60)
                # for callsign, positions in group.groupby(callsign_col):
                #     positions.set_index('t')
                #     positions = positions[~positions.index.duplicated()]
                #     positions_resampled = positions.set_index('t').resample('min')
                #
                # if cluster != -1:
                #     pass
            flows = CombinedFlows(OrderedDict(flows_dict))

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
        mask = np.ones_like(between_flow_conflicts)

        mask[between_flow_conflicts > 0] = False
        mask[0,:] = True
        sns.heatmap(between_flow_conflicts, mask=mask, cmap="flare", ax=ax[split])

        heatmap_list.append(between_flow_conflicts)
    plt.show()
    df_results = pd.DataFrame(records)
    plt.figure()
    df_results.boxplot(column='conflicts', by=['type', 'unclustered'])
    plt.show()
    plt.figure()
    df_results.query('unclustered == False').boxplot(column='conflicts', by=['split', 'type'])
    plt.show()
    df_results.to_excel(results_fn)