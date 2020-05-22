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
    lognames = {'CONFLOG_val_uniform_20200522_09-42-25.log': {'IV': 'heading', 'values': (headings := [0, 30, 60, 90, 120, 150, 180]), 'map': lambda i, j: np.abs(headings[i] - headings[j]), 'N_per_IV': 40, 'label': 'Varying heading, spawn time pdf: U(400,600)s','subplot': 121, 'C': 'C0'},
                'CONFLOG_val_uniform_20200522_11-55-48.log': {'IV': 'heading', 'values': (headings := [0, 30, 60, 90, 120, 150, 180]), 'map': lambda i, j: np.abs(headings[i] - headings[j]), 'N_per_IV': 40, 'label': 'Varying heading, spawn time pdf: U(60,600)s', 'subplot': 122, 'C': 'C1'},
                }

    records = []
    dfs = []
    for logname, options in lognames.items():
        conflict_counts = get_conflict_counts_from_logfile(logname, logdir, dt_before_new_conflict=99999, minimim_duration_for_conflict=None)
        conflicts_by_cluster = inter_and_intra_cluster_conflicts(conflict_counts)
        N = options['N_per_IV']
        # ax = plt.subplot(options['subplot'])
        df = pd.DataFrame([(options['map'](i, j), conflicts_by_cluster[i, j]) for i in range(7) for j in range(7)],
                          columns=['IV', 'conflicts'])
        # df.plot.scatter(x='IV', y='conflicts', ax=ax, label=options['label'])
        df['label'] = options['label']
        df['IV_name'] = options['IV']
        dfs.append(df)

    plt.legend()
    plt.show()

    df_combined = pd.concat(dfs)
    fix, ax = plt.subplots()
    plot_i = 0

    for name, group in df_combined.groupby('label'):
        group.plot.scatter(x='IV', y='conflicts', ax=ax, label=name, c="C{}".format(plot_i), marker=str((plot_i + 1) % 4))
        y_data = group['conflicts'].values.reshape(-1, 1)
        x_data = group['IV'].values.reshape(-1, 1)
        reg = linear_model.LinearRegression()
        reg.fit(x_data, y_data)
        ax.plot(x_data, reg.predict(x_data), '--', c="C{}".format(plot_i))
        if len(IV_names := group['IV'].unique()) == 1:
            ax.xlabel(IV_names[0])
        plot_i += 1
    plt.legend()
    plt.show()


