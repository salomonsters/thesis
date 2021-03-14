import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import numpy as np


if __name__ == "__main__":
    filter = ''
    # out_filenames = ['data/simulated_conflicts/gs-80_trk-0-1-360_vs-0.xlsx',
    #           'data/simulated_conflicts/gs-100_trk-0-1-360_vs-0.xlsx',
    #           'data/simulated_conflicts/gs-120_trk-0-1-360_vs-0.xlsx']
    # fig, ax = plt.subplots(figsize=(10,6))
    # df_list = []
    # for i, fn in enumerate(out_filenames):
    #     df = pd.read_excel(fn)
    #     df['fn'] = fn.split('/')[-1].split('.')[0]
    #     df['x'] = df['IV'] + i
    #     df['y_inv'] = 1/df['y']
    #     df['color'] = 'C{}'.format(i)
    #     df_list.append(df)
    # dfs = pd.concat(df_list)

    # for name, group in dfs.groupby('fn'):
    #     group.plot(kind='scatter', x='x', y='y_inv', ax=ax, label=name, c=group['color'], s=0.1)
    # lgnd = plt.legend(scatterpoints=1, fontsize=10)
    # for i in range(len(lgnd.legendHandles)):
    #     lgnd.legendHandles[i]._sizes = [30]
    # plt.show()

    out_filenames = [# 'data/simulated_conflicts/poisson-gs-80-100-120_trk-0-1-360_vs-0.xlsx',
                    # 'data/simulated_conflicts/poisson-f-3600-gs-100_trk-0-1-360_vs-0.xlsx',
                    # 'data/simulated_conflicts/poisson-choice-f-3600-gs-100_trk-0-1-360_vs-0.xlsx'
                    # 'data/simulated_conflicts/poisson-nochoice-f-3600-gs-100_trk-0-1-360_vs-0.xlsx'
                    # 'data/simulated_conflicts/poisson-nochoice-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8nm.xlsx'
                    # 'data/simulated_conflicts/poisson-nochoice-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8.5nm.xlsx'
                    # 'data/simulated_conflicts/poisson-25percentdeviation-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8.5nm.xlsx'
        # 'data/simulated_conflicts/poisson-25percentdeviation-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8.5nm-measured-spawndistances.xlsx'
       # 'data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0.xlsx',
       # 'data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0-lam_based_on_V_exp_200.xlsx',
        'data/simulated_conflicts/poisson-f-3600-gs-200-10-400_trk-0-30-120_vs-0-lam_based_on_V_exp_200.xlsx',
        'data/simulated_conflicts/poisson-f-3600-gs-200-10-400_trk-0-30-120_vs-0-lam_based_on_V_exp_200-realisation-2.xlsx'
                     ]
    # T = 3
    # ax_twin = ax[0].twinx()

    df_list = []
    f_simulation = None
    from numpy import array
    for i, fn in enumerate(out_filenames):
        IV_query = None
        df = pd.read_excel(fn)
        df = df[~pd.isna(df['flow2'])]
        df['fn'] = fn.split('/')[-1].split('.')[0]
        df['other_properties'] = df['other_properties'].str.replace('\n', '').str.replace('array([', '[',
                                                                                          regex=False).str.replace('])',
                                                                                                                   ']',
                                                                                                                   regex=False).apply(
            ast.literal_eval)
        df['other_properties_flow2'] = df['other_properties_flow2'].str.replace('\n', '').str.replace('array([', '[',
                                                                                                      regex=False).str.replace(
            '])', ']', regex=False).apply(ast.literal_eval)
        if 'IV' in df['other_properties'].iloc[0]:
            IV = df['other_properties'].iloc[0]['IV']
            if IV == 'gs':
                df['trk_diff_uncorrected'] = np.abs(np.mod(df['trk_flow2'] - df['trk'], 360))
                df['trk_diff'] = np.where(df['trk_diff_uncorrected'] > 180, 360 - df['trk_diff_uncorrected'], df['trk_diff_uncorrected'])
                df['x'] = np.abs(df[IV] - df[IV + '_flow2'])
                IV_query = 'trk_diff>1'
                IV_fancy_name = 'Groundspeed difference (kts)'
            else:
                raise NotImplementedError()

        else:
            df['x'] = df['IV'] # + (df['gs'].astype(float) - 100)/20
            df['trk_diff'] = np.abs(np.mod(df['trk_flow2'] - df['trk'], 360))
            df['trk_diff'][df['trk_diff'] > 180] = 360 - df['trk_diff'][df['trk_diff'] > 180]
            IV = 'trk_diff'
            IV_query = 'abs_sin_rad_trk_diff>0.01'
            IV_fancy_name = 'Angle between flows (degree)'
        df['y_inv'] = 1/df['y']
        # df['color'] = 'C' + (((df['gs'].astype(float) - 100)/20).astype(int)+1).astype(str)

        # df['other_properties_flow2'] = df['other_properties_flow2'].apply(lambda x: ast.literal_eval if ~pd.isna(x) else defaultdict(lambda: np.nan))
        # df[~pd.isna(df['other_properties_flow2'])]['other_properties_flow2'] = df[~pd.isna(df['other_properties_flow2'])]['other_properties_flow2'].apply(ast.literal_eval)
        # df[ pd.isna(df['other_properties_flow2'])]['other_properties_flow2'] = defaultdict(lambda: np.nan)
        df_list.append(df)
        f = None
        for k, part in enumerate(fn.split('-')):
            if part == 'f':
                f = fn.split('-')[k+1]
        if f is None or (f_simulation is not None and float(f) != f_simulation):
            raise RuntimeError("We have no simulation frequency")
        else:
            f_simulation = float(f)
    dfs = pd.concat(df_list)# .reset_index(drop=True)

    # Only look at conflicts between flows:

    dfs['abs_trk_diff_in_rad'] = np.abs(np.radians((dfs['trk_flow2']-dfs['trk'] + 360)%360))
    dfs['Vr,h'] = (dfs['gs'] ** 2 + dfs['gs_flow2'] ** 2 - 2 * dfs['gs'] * dfs['gs_flow2'] * np.cos(
        dfs['abs_trk_diff_in_rad'])) ** 0.5

    dfs['abs_sin_rad_trk_diff'] = np.sin(dfs['abs_trk_diff_in_rad'])


    from conflicts.simulate import Aircraft
    def calculate_correction_factor(B1_exp, B2_exp):
        g = Aircraft.horizontal_separation_requirement
        # h = Aircraft.vertical_separation_requirement
        # b = Aircraft.vertical_separation_requirement

        # return 4 * g * h / (b * B1_exp * B2_exp)
        return 2*g / ( B1_exp * B2_exp)

    # T_exp = n_aircraft_per_flow * horizontal_distance_exp / V_exp
    dfs['conflictsph'] = dfs['y']# /T_exp
    dfs['lam'] = pd.json_normalize(dfs['other_properties'])['lam']
    dfs['lam_flow2'] = pd.json_normalize(dfs['other_properties_flow2'])['lam']
    # try:
    #     B1_exp = pd.json_normalize(dfs['other_properties'])['measured_distances_at_spawn'].apply(np.mean)
    #     B2_exp = pd.json_normalize(dfs['other_properties_flow2'])['measured_distances_at_spawn'].apply(np.mean)
    # except:
    B1_exp = dfs['gs'] / (dfs['lam'] * f_simulation)
    B2_exp = dfs['gs_flow2'] / (dfs['lam_flow2'] * f_simulation)

    # B1_exp = horizontal_distance_exp
    # B2_exp = horizontal_distance_exp

    correction_factor = calculate_correction_factor(B1_exp, B2_exp)
    dfs_ungrouped = dfs
    # for g, dfs in dfs_ungrouped.groupby(['lam', 'lam_flow2']):

    if True:
        fig, ax = plt.subplots(5, 1, figsize=(10, 12))

        # dfs['conflicts_predicted'] = (dfs['Vr,h']/(dfs['gs'] * dfs['gs_flow2']*np.sin(dfs['abs_trk_diff_in_rad']))) * correction_factor
        dfs['conflicts_predicted'] = (dfs['Vr,h']/(np.sin(dfs['abs_trk_diff_in_rad']))) * correction_factor
        if IV_query is not None:
            dfs.query(IV_query, inplace=True)
        dfs.plot(kind='scatter', x='x', y='conflictsph', ax=ax[0], label="Observed conflict rate", c='C0', s=0.3)
        ax[0].set_xlabel(IV_fancy_name)
        ylim_observed = ax[0].get_ylim()


        ax[0].legend(loc='upper left')
        dfs.plot(kind='scatter', x='x', y='conflicts_predicted', ax=ax[0], label="Predicted conflict rate", c='C1', s=0.3)
        ax[0].set_ylim(ylim_observed)
        # ax_twin.legend(loc='upper right')
        # ax_twin.set_xlabel("Angle between flows in deg")

        ax[0].set_ylabel("Conflict rate")
        # ax_twin.set_ylabel("Predicted conflicts")
        dfs.plot(kind='scatter', x='conflicts_predicted', y='conflictsph', ax=ax[1], label="Observed Conflict rate", c='C0', s=0.3)
        ax[1].set_xlabel("Predicted conflict rate")
        ax[1].set_ylabel("Observed conflict rate")

        dfs.plot(kind='scatter', x='conflicts_predicted', y='x', ax=ax[2], label="Prediction", c='C1', s=0.3)
        ax[2].set_ylabel(IV_fancy_name)

        dfs['observed/predicted'] = dfs['conflictsph']/dfs['conflicts_predicted']

        dfs.plot(kind='scatter', x='x', y='observed/predicted', ax=ax[3], c='C1', s=0.3)

        dfs.boxplot(column='observed/predicted', by='x', ax=ax[4])
        ax[4].set_title('')
        ax[4].set_ylabel('observed/predicted')
        ax[4].set_xlabel(IV_fancy_name)
        import matplotlib.ticker as ticker
        ax[4].xaxis.set_major_locator(ticker.MultipleLocator(10))
        # dfs.boxplot sets the figure title but we do not want that
        fig.suptitle('')


        # ax4_twin = ax[4].twinx()

        all_samples = dfs.groupby('x')['observed/predicted'].apply(lambda x: x.tolist()).to_dict()
        from scipy import stats
        df_ks2 = pd.DataFrame(
            [(k, stats.ks_2samp(all_samples[k], dfs['observed/predicted']).pvalue) for k in all_samples.keys()],
            columns=['IV', 'p'])
        df_ks2['significantly_different_distribution_alpha_0.01'] = (df_ks2['p'] < 0.01).astype(int)
        # df_ks2.plot.scatter(x='angle', y='significantly_different_distribution_alpha_0.01', ax=ax4_twin)
        # plt.show()




        # lgnd = plt.legend(scatterpoints=1, fontsize=10)
        # for i in range(len(lgnd.legendHandles)):
        #     lgnd.legendHandles[i]._sizes = [30]
        plt.show()
        # print(g)
        # input("press enter to continue")