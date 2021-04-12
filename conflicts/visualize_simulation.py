import ast

import matplotlib
import matplotlib.offsetbox
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression

from conflicts.simulate import Aircraft


def plot_simulation_consistency(input_filenames, plot_options=None):
    df_list = []
    f_simulation = None
    IV_query = None
    IV_plot_callback = lambda fig, ax: None
    IV = None
    for i, fn in enumerate(input_filenames):
        df = pd.read_excel(fn)
        df = df[~pd.isna(df['flow2'])]
        df.reset_index(inplace=True, drop=True)
        df['fn'] = fn.split('/')[-1].split('.')[0]
        df['other_properties'] = df['other_properties']. \
            str.replace('\n', '').str.replace('array([', '[', regex=False).str.replace('])', ']', regex=False).apply(
            ast.literal_eval)
        df['other_properties_flow2'] = df['other_properties_flow2']. \
            str.replace('\n', '').str.replace('array([', '[', regex=False).str.replace('])', ']', regex=False).apply(
            ast.literal_eval)
        f = None
        for k, part in enumerate(fn.split('-')):
            if part == 'f':
                f = fn.split('-')[k + 1]
        if f is None or (f_simulation is not None and float(f) != f_simulation):
            raise RuntimeError("We have no simulation frequency")
        else:
            if f_simulation is None:
                f_simulation = float(f)
            else:
                # Not the first file
                if float(f) != f_simulation:
                    raise ValueError("Simulation frequencies {} and {} don't match".format(float(f), f_simulation))

        if 'IV' in df['other_properties'].iloc[0]:
            if IV is not None and IV != df['other_properties'].iloc[0]['IV']:
                raise ValueError("IV's {} and {} not of same type".format(IV, df['other_properties'].iloc[0]['IV']))
            IV = df['other_properties'].iloc[0]['IV']
            if IV == 'gs':
                df['trk_diff_uncorrected'] = np.abs(np.mod(df['trk_flow2'] - df['trk'], 360))
                df['trk_diff'] = np.where(df['trk_diff_uncorrected'] > 180, 360 - df['trk_diff_uncorrected'],
                                          df['trk_diff_uncorrected'])
                df['x'] = (df[IV] - df[IV + '_flow2']).apply(lambda x: round(np.abs(x)))
                IV_query = 'trk_diff>1'
                IV_fancy_name = 'Groundspeed difference [kts]'
            elif IV == 'lam':
                df['trk_diff_uncorrected'] = np.abs(np.mod(df['trk_flow2'] - df['trk'], 360))
                df['trk_diff'] = np.where(df['trk_diff_uncorrected'] > 180, 360 - df['trk_diff_uncorrected'],
                                          df['trk_diff_uncorrected'])
                df['lam'] = pd.json_normalize(df['other_properties'])['lam']
                df['lam_flow2'] = pd.json_normalize(df['other_properties_flow2'])['lam']
                # df['lam'] is in aircraft/timestep, so need to multiply with f_simulation to get aircraft/hr
                df['x'] = (f_simulation * (df['lam'] + df['lam_flow2']) / 2).apply(round)
                IV_fancy_name = 'Mean arrival rate $\lambda$ [aircraft/hr]'
            elif IV == 'trk_diff':
                df['trk_diff_uncorrected'] = np.abs(np.mod(df['trk_flow2'] - df['trk'], 360))
                df['trk_diff'] = np.where(df['trk_diff_uncorrected'] > 180, 360 - df['trk_diff_uncorrected'],
                                          df['trk_diff_uncorrected'])
                df['x'] = (10 * np.round(df['trk_diff']/10.)).apply(round)
                IV_query = 'abs_sin_rad_trk_diff>0.01'
                IV_fancy_name = 'Angle between flows [$^\circ$]'
                # IV_plot_callback = lambda fig, ax: ax[1].xaxis.set_major_locator(ticker.MultipleLocator(10))
            else:
                raise NotImplementedError()
        else:
            raise NotImplementedError("Need to specify IV_type in other_properties")
        import warnings
        if 'S_h' in df['other_properties'].iloc[0]:
            df['S_h'] = df['other_properties'].iloc[0]['S_h']
            warnings.warn("Using S_h={}".format(df['other_properties'].iloc[0]['S_h']))
        else:
            df['S_h'] = Aircraft.horizontal_separation_requirement
            warnings.warn("Using default separation requirement of {}".format(Aircraft.horizontal_separation_requirement))

        df_list.append(df)

    dfs = pd.concat(df_list).reset_index(drop=True)

    # Only look at conflicts between flows:

    dfs['abs_trk_diff_in_rad'] = np.abs(np.radians((dfs['trk_flow2'] - dfs['trk'] + 360) % 360))
    dfs['Vr,h'] = (dfs['gs'] ** 2 + dfs['gs_flow2'] ** 2 - 2 * dfs['gs'] * dfs['gs_flow2'] * np.cos(
        dfs['abs_trk_diff_in_rad'])) ** 0.5

    dfs['abs_sin_rad_trk_diff'] = np.sin(dfs['abs_trk_diff_in_rad'])

    dfs['conflictsph'] = dfs['y']  # /T_exp

    dfs['lam'] = pd.json_normalize(dfs['other_properties'])['lam']
    dfs['lam_flow2'] = pd.json_normalize(dfs['other_properties_flow2'])['lam']
    B1_exp = dfs['gs'] / (dfs['lam'] * f_simulation)
    B2_exp = dfs['gs_flow2'] / (dfs['lam_flow2'] * f_simulation)
    correction_factor = 2 * dfs['S_h'] / (B1_exp * B2_exp)
    dfs['conflicts_predicted'] = (dfs['Vr,h'] / (np.sin(dfs['abs_trk_diff_in_rad']))) * correction_factor
    dfs['observed/predicted'] = dfs['conflictsph'] / dfs['conflicts_predicted']

    if IV_query is not None:
        dfs.query(IV_query, inplace=True)

    fig, ax = plt.subplots(2, 1, figsize=(5, 10))

    dfs.plot(kind='scatter', x='conflicts_predicted', y='conflictsph', ax=ax[0], c='C0', s=0.3, label="Simulations")
    ax[0].set_xlabel("Predicted conflict rate [1/hr]")
    ax[0].set_ylabel("Observed conflict rate [1/hr]")

    ax0_xlims = ax[0].get_xlim()
    ax0_ylims = ax[0].get_ylim()

    ax0_lims = [
        np.min([ax[0].get_xlim(), ax[0].get_ylim()]),  # min of both ax[0]es
        np.max([ax[0].get_xlim(), ax[0].get_ylim()]),  # max[0] of both ax[0]es
    ]

    x_col = 'conflicts_predicted'
    y_col = 'conflictsph'
    X = dfs[x_col].values.reshape(-1, 1)
    Y = dfs[y_col].values.reshape(-1, 1)
    non_nan_values = ~pd.isna(X + Y) & np.isfinite(X + Y) & (Y > 0.01) #& (X < 10) & (Y < 20)
    linear_regressor = LinearRegression()
    linear_regressor.fit(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1))
    r_squared = linear_regressor.score(X[non_nan_values].reshape(-1, 1), Y[non_nan_values].reshape(-1, 1))

    # now plot both limits against eachother
    ax[0].plot(ax0_lims, ax0_lims, 'k--', label="Theoretical", alpha=0.75, zorder=0)
    ax[0].set_xlim(ax0_xlims)
    ax[0].set_ylim(ax0_ylims)
    anchored_text = matplotlib.offsetbox.AnchoredText("$R^2={:.4f}$".format(r_squared), loc="lower right")
    ax[0].add_artist(anchored_text)
    ax[0].legend()

    dfs.boxplot(column='observed/predicted', by='x', ax=ax[1], rot=90)
    ax[1].set_title('')
    ax[1].set_ylabel('Observed/Predicted')
    ax[1].set_xlabel(IV_fancy_name)
    ax[1].axhline(y=1)
    IV_plot_callback(fig, ax)
    # dfs.boxplot sets the figure title but we do not want that
    fig.suptitle('')

    if plot_options is not None:
        plot_type = plot_options['fn'].rsplit('.')[-1]
        if plot_type == 'pgf':
            matplotlib.use("pgf")
            matplotlib.rcParams.update({
                "pgf.texsystem": "pdflatex",
                'font.family': 'serif',
                'text.usetex': True,
                'pgf.rcfonts': False,
            })
        fig.set_size_inches(**plot_options['size'])
        fig.savefig(plot_options['fn'], bbox_inches='tight')
        print("Saved output to {}".format(plot_options['fn']))

        if plot_type != "pgf":
            plt.show()

    # Return dfs for possible usage
    return dfs


if __name__ == "__main__":
    out_filenames = [  # 'data/simulated_conflicts/poisson-gs-80-100-120_trk-0-1-360_vs-0.xlsx',
        # 'data/simulated_conflicts/poisson-f-3600-gs-100_trk-0-1-360_vs-0.xlsx',
        # 'data/simulated_conflicts/poisson-choice-f-3600-gs-100_trk-0-1-360_vs-0.xlsx'
        # 'data/simulated_conflicts/poisson-nochoice-f-3600-gs-100_trk-0-1-360_vs-0.xlsx'
        # 'data/simulated_conflicts/poisson-nochoice-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8nm.xlsx'
        # 'data/simulated_conflicts/poisson-nochoice-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8.5nm.xlsx'
        # 'data/simulated_conflicts/poisson-25percentdeviation-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8.5nm.xlsx'
        # 'data/simulated_conflicts/poisson-25percentdeviation-f-3600-gs-100_trk-0-1-360_vs-0-intended_sep-8.5nm-measured-spawndistances.xlsx'
        # 'data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0.xlsx',
        # 'data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0-lam_based_on_V_exp_200.xlsx',
        #  'data/simulated_conflicts/poisson-f-3600-gs-200-10-400_trk-0-30-120_vs-0-lam_based_on_V_exp_200.xlsx',
        # 'data/simulated_conflicts/poisson-f-3600-gs-200-10-400_trk-0-30-120_vs-0-lam_based_on_V_exp_200-realisation-2.xlsx',
        # 'data/simulated_conflicts/poisson-f-3600-gs-200-10-400_trk-0-30-120_vs-0-lam_based_on_V_exp_200-S_h-5nm.xlsx',
        # 'data/simulated_conflicts/poisson-f-3600-gs-200_trk-0-30-120_vs-0-lam--1e-3--1e-3--1e-2.xlsx'
    ]
    # out_filenames = ['data/simulated_conflicts/'
    #                  'poisson-f-3600-gs-200_trk-0-30-120_vs-0-acph-3-2-18-realisation-{}.xlsx'.format(realisation)
    #                  for realisation in range(10)]
    # out_filenames = [ 'data/simulated_conflicts/' \
    #              'poisson-f-3600-gs-{}_trk-0-2.5-360_vs-0-acph-{}-realisation-{}.xlsx'.format(200, 15, realisation) for realisation in range(10)]
    out_filenames = [ 'data/simulated_conflicts/' \
                 'poisson-f-3600-gs-{}_trk-0-2.5-360_vs-0-acph-{}-R_{}nm-realisation-{}.xlsx'.format(200, 15, 200, r) for r in range(2)]
    out_filenames = ['data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0-acph-15-R_200nm-realisation-0.xlsx',
                     'data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0-acph-15-R_200nm-realisation-1.xlsx',
    'data/simulated_conflicts/poisson-f-3600-gs-100-5-200_trk-0-30-120_vs-0-acph-15-R_200nm-realisation-2.xlsx']
    plot_fn = 'pgf/varying_isolated_quantities/vary_trk_0_2.5_360-gs-{}-{}_realisations-{}_acph-R_{}nm.pgf'.format(
        200, 200, 15, 200)
    plot_options = {
        'fn': plot_fn,
        'size': {'w': 3.5, 'h': 6}
    }
    plot_options = None
    dfs = plot_simulation_consistency(out_filenames, plot_options=plot_options)
