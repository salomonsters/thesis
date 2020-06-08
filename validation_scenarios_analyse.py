import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

from conflog_analyse import unique_conflicts, get_conflict_counts_from_logfile, inter_and_intra_cluster_conflicts

if __name__ == "__main__":
    logdir = '../bluesky/output'
    logfiles = [{'filename': 'CONFLOG_val_uniform_20200522_09-42-25.log', 'IV': 'heading', 'values': (headings := [0, 30, 60, 90, 120, 150, 180]), 'map': lambda i, j: np.abs(headings[i] - headings[j]), 'N_per_IV': 40, 'regression_exclude': "IV == 0", 'label': 'Varying heading, spawn time pdf: U(400,600)s'},
                {'filename': 'CONFLOG_val_uniform_20200522_11-55-48.log', 'IV': 'heading', 'values': (headings := [0, 30, 60, 90, 120, 150, 180]), 'map': lambda i, j: np.abs(headings[i] - headings[j]), 'N_per_IV': 40, 'regression_exclude': "IV == 0", 'label': 'Varying heading, spawn time pdf: U(60,600)s' },
                # the i+j%2==1 is there to only count flows with a 90* heading wrt each other
                {'filename': 'CONFLOG_velocity_U60-80_U80-100_U-100-120_heading_C0_90_spawn_U400-600_20200522_14-18-36.log', 'IV': 'gs', 'values': (mean_speeds := [70, 70, 90, 90, 110, 110]), 'map': lambda i, j: np.abs(mean_speeds[i] - mean_speeds[j]) if (i+j)%2 == 1 else np.nan, 'N_per_IV': 40, 'label': 'Varying speed: uniform(kts +/- 10) , spawn time pdf: U(60,600)s, 90 deg heading offset'},
                {'filename': 'CONFLOG_velocity_U60-80_U80-100_U-100-120_heading_C0_90_spawn_U400-600_20200522_14-18-36.log', 'IV': 'gs', 'values': (mean_speeds := [70, 70, 90, 90, 110, 110]), 'map': lambda i, j: 0 if i == j else np.nan, 'N_per_IV': 40, 'label': 'Varying speed: uniform(kts +/- 10) , spawn time pdf: U(60,600)s, same heading'},
                {'filename': 'CONFLOG_velocity_U60-80_U80-100_U-100-120_heading_C0_90_spawn_U400-600_20200525_15-25-51.log', 'IV': 'gs', 'values': (mean_speeds := [70, 70, 90, 90, 110, 110]), 'map': lambda i, j: 0 if i == j else np.nan, 'N_per_IV': 40, 'label': 'Varying speed: uniform(kts +/- 10) , spawn time pdf: U(60,600)s, same heading',
                    'IV_transform': lambda df_conflicts: (df_conflicts['ac1'].str.split('_').str[2].astype(int)-df_conflicts['ac2'].str.split('_').str[2].astype(int))},
                ]

    records = []
    dfs = []
    for options in logfiles:
        logname = options['filename']
        conflict_counts, df_conflicts = get_conflict_counts_from_logfile(logname, logdir, dt_before_new_conflict=99999, minimim_duration_for_conflict=None, return_df_conflicts=True)
        conflicts_by_cluster = inter_and_intra_cluster_conflicts(conflict_counts)
        # N = options['N_per_IV']
        # ax = plt.subplot(options['subplot'])
        if 'IV_transform' in options:
            df_conflicts['IV'] = options['IV_transform'](df_conflicts)
            df_conflicts['ac_pair'] = df_conflicts.apply(lambda r: (r['ac1'], r['ac2']), axis=1)
            IV = df_conflicts.groupby('ac_pair')['IV'].unique().apply(np.asscalar)

            df = pd.DataFrame([(options['map'](i, j), conflicts_by_cluster[i, j]) for i in range(n_clusters) for j in
                               range(n_clusters)],
                              columns=['IV', 'conflicts'])
        n_clusters = len(options['values'])
        df = pd.DataFrame([(options['map'](i, j), conflicts_by_cluster[i, j]) for i in range(n_clusters) for j in range(n_clusters)],
                          columns=['IV', 'conflicts'])
        df['regression_exclude'] = False
        if 'regression_exclude' in options and options['regression_exclude'] is not None:
            df['regression_exclude'].iloc[df.query(options['regression_exclude']).index] = True
        # df.plot.scatter(x='IV', y='conflicts', ax=ax, label=options['label'])
        df['label'] = options['label']
        df['IV_name'] = options['IV']
        dfs.append(df)

    # plt.legend()
    # plt.show()

    df_combined = pd.concat(dfs)

    for IV_name, group_IV in df_combined.groupby('IV_name'):
        fix, ax = plt.subplots(figsize=(8, 6))
        plot_i = 0
        for name, group in group_IV.groupby('label'):
            group.plot.scatter(x='IV', y='conflicts', ax=ax, label=name, c="C{}".format(plot_i), marker=str((plot_i + 1) % 4))
            regression_data = group
            y_data = group['conflicts'].values.reshape(-1, 1)
            x_data = group['IV'].values.reshape(-1, 1)
            reg = linear_model.LinearRegression()
            selector = ~np.isnan(x_data) & ~np.isnan(y_data) & ~group['regression_exclude'].values.reshape(-1, 1)
            reg.fit(x_data[selector].reshape(-1, 1), y_data[selector].reshape(-1, 1))
            ax.plot(x_data[selector].reshape(-1, 1), reg.predict(x_data[selector].reshape(-1, 1)), '--', c="C{}".format(plot_i))
            ax.set_xlabel(IV_name)
            plot_i += 1
        plt.legend()
        plt.show()


# (df_conflicts['ac1'].str.split('_').str[2].astype(int)-df_conflicts['ac2'].str.split('_').str[2].astype(int)).plot.kde()