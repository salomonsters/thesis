import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

if __name__ == "__main__":
    filter = ''
    out_filenames = ['data/simulated_conflicts/gs-80_trk-0-1-360_vs-0.xlsx',
              'data/simulated_conflicts/gs-100_trk-0-1-360_vs-0.xlsx',
              'data/simulated_conflicts/gs-120_trk-0-1-360_vs-0.xlsx']
    fig, ax = plt.subplots(figsize=(10,6))
    df_list = []
    for i, fn in enumerate(out_filenames):
        df = pd.read_excel(fn)
        df['fn'] = fn.split('/')[-1].split('.')[0]
        df['x'] = df['IV'] + i
        df['y_inv'] = 1/df['y']
        df['color'] = 'C{}'.format(i)
        df_list.append(df)
    dfs = pd.concat(df_list)

    for name, group in dfs.groupby('fn'):
        group.plot(kind='scatter', x='x', y='y_inv', ax=ax, label=name, c=group['color'], s=0.1)
    lgnd = plt.legend(scatterpoints=1, fontsize=10)
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [30]
    plt.show()

    out_filenames = ['data/simulated_conflicts/gs-80-100-120_trk-0-1-360_vs-0.xlsx']
    fig, ax = plt.subplots(3, 1, figsize=(10,12))

    df_list = []
    for i, fn in enumerate(out_filenames):
        df = pd.read_excel(fn)
        df['fn'] = fn.split('/')[-1].split('.')[0]
        df['x'] = df['IV'] + (df['gs'].astype(float) - 100)/20
        df['y_inv'] = 1/df['y']
        df['color'] = 'C' + (((df['gs'].astype(float) - 100)/20).astype(int)+1).astype(str)
        df_list.append(df)
    dfs = pd.concat(df_list)
    dfs['abs_trk_diff_in_rad'] = np.abs(np.radians(dfs['trk_flow2']-dfs['trk']))
    dfs['Vr,h'] = (dfs['gs'] ** 2 + dfs['gs_flow2'] ** 2 - 2 * dfs['gs'] * dfs['gs_flow2'] * np.cos(
        np.radians(dfs['trk_flow2'] - dfs['trk']))) ** 0.5
    dfs['C'] = (dfs['Vr,h']/(dfs['gs'] * dfs['gs_flow2']*np.sin(np.radians(dfs['trk_flow2'] - dfs['trk']))))
    from math import radians
    dfs['abs_sin_rad_trk_diff'] = np.sin(dfs['abs_trk_diff_in_rad'])
    dfs.query('abs_sin_rad_trk_diff>0.01', inplace=True)
    for name, group in dfs.groupby(['gs', 'gs_flow2']):
        group.plot(kind='scatter', x='x', y='y', ax=ax[0], label="gs-{}".format(name), c=group['color'], s=0.3)
        ax0_2 = ax[0].twinx()
        group.plot(kind='scatter', x='x', y='C', ax=ax[1], label="gs-{}".format(name), c=group['color'], s=0.3)
        # group.plot(kind='scatter', x='x', y='y_inv', ax=ax[0], label="gs-{}".format(name), c=group['color'], s=0.3)
        # group.plot(kind='scatter', x='Vr,h', y='y_inv', ax=ax[1], label="gs-{}".format(name), c=group['color'], s=0.3)

    lgnd = plt.legend(scatterpoints=1, fontsize=10)
    for i in range(len(lgnd.legendHandles)):
        lgnd.legendHandles[i]._sizes = [30]
    plt.show()