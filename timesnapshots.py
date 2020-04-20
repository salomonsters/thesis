import numpy as np
import copy
import math
from itertools import cycle
from scipy.linalg import eigh
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import matplotlib.colors
import numba
from numba import cuda
import glob, os
from tools import create_logger
from clustering import lat2y, lon2x
from sklearn.cluster import DBSCAN, OPTICS
import scipy.sparse as ss
from scipy.spatial.distance import pdist, cdist


def pairwise_range(X):
    """
    Find pairwise range of 2d matrix with x, y locations.
    :param X: 2d matrix with x, y locations.
    :return: condensed distance matrix Y
    """
    return pdist(X)


def range_rate(A, B):
    R = B[:2] - A[:2]
    vA = np.array([A[2]*math.sin(np.deg2rad(A[3])), A[2]*math.cos(np.deg2rad(A[3]))])
    vB = np.array([B[2]*math.sin(np.deg2rad(B[3])), B[2]*math.cos(np.deg2rad(B[3]))])
    return 1/np.linalg.norm(R)*np.inner(R, vA - vB)


def pairwise_range_rate(X):
    """
    Find pairwise range rate of 2d matrix with fields [x, y, gs, trk]
    :param X: 2d matrix with fields [x, y, gs, trk]
    :return: condensed pairwise range rate matrix Y
    """
    return pdist(X, range_rate)


def cut_interval(df, dt, timeCol='ts'):

    t0 = df[timeCol].min()
    tend = df[timeCol].max()

    bins = pd.interval_range(start=t0, end=tend+dt, freq=dt, closed='left')
    df['interval'] = pd.cut(df[timeCol], bins)
    return df.set_index('interval', drop=True)


if __name__ == "__main__":
    verbose = True
    airport = 'EHAM'
    airspace_query = "airport=='{}'".format(airport)
    zoom = 15
    minalt = 200  # ft
    maxalt = 10000

    fields = ['lat', 'lon']

    airspace = ehaa_airspace.query(airspace_query)
    airspace_x = lon2x(airspace[airspace['type'] == 'CTR'].geometry.centroid.x).iloc[0]
    airspace_y = lat2y(airspace[airspace['type'] == 'CTR'].geometry.centroid.y).iloc[0]

    log = create_logger(verbose, "Statistics")


    log("Start reading csv")

    # #### SINGLE FILENAME
    df = pd.read_csv('data/adsb_decoded_in_{0}/ADSB_DECODED_20180101.csv.gz'.format(airport))

    # #### ENTIRE DIRECTORY
    # path = './data/adsb_decoded_in_eham/'
    # max_number_of_files = 20
    # df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))[:max_number_of_files]))
    log("Completed reading CSV")

    orig_df = copy.deepcopy(df)

    if minalt is not None:
        df = df[df['alt'] > minalt]
    if maxalt is not None:
        df = df[df['alt'] < maxalt]

    df['x'] = lon2x(df['lon'].values)
    df['y'] = lat2y(df['lat'].values)

    df.sort_values(by='ts', inplace=True)

    # Select time increment
    delta_t = 5*60 # seconds
    time_tolerance = 2 # seconds
    grouped_per_interval = cut_interval(df, dt=delta_t).groupby('interval')
    result_records = []
    ranges_list = []
    range_rates_list = []
    for interval, interval_group in grouped_per_interval:
        if len(interval_group) == 0:
            log("Skipping empty interval {0}".format(interval))
            continue
        fid_per_interval = interval_group.groupby('fid')
        min_times = fid_per_interval['ts'].min() - interval.left

        # Find fid's that have their first data within time_tolerance seconds of the lowest ts in the interval
        fids_for_snapshot = min_times[min_times < min_times.min() + time_tolerance].index.values
        # Get the first row of each groupby object
        snapshot = fid_per_interval.nth(0).loc[fids_for_snapshot]
        if len(snapshot) == 0:
            log("Skipping empty interval {0}".format(interval))
            continue
        # Check that timestamps are still within tolerance
        times = snapshot['ts'].values
        if np.max(times[:,None]-times) > time_tolerance:
            raise Exception("Time tolerance out of bounds, maybe dataframe is not sorted?")
        # Find pairwise distances (range)
        ranges = pairwise_range(snapshot[['x', 'y']])
        range_rates = pairwise_range_rate(snapshot[['x', 'y', 'gs', 'trk']])
        if len(ranges) > 0:
            # ranges_list.append(interval.left)
            ranges_list.append(ranges)
            range_rates_list.append(range_rates)

        for i in range(len(ranges)):
            result_records.append((interval.left, ranges[i], range_rates[i]))

    log("Finished generating snapshots")

    result_df = pd.DataFrame.from_records(result_records, columns=['interval_left', 'range', 'range_rate'])
    # result_df.query('{0} < interval_left < ({0} + {1})'.format(result_df['interval_left'].median(), 3600), inplace=True)
    # fig1, ax1 = plt.subplots(figsize=(10, 4))
    # result_df.plot.scatter(x='interval_left', y='range', ax=ax1, s=0.3)
    # ax2 = ax1.twinx()
    # result_df.plot.scatter(x='interval_left', y='range_rate', ax=ax2, c='r', s=0.3)
    # # ax2.scatter(result_df['interval_left'], result_df['range_rate'])
    # plt.show()
    fig, ax1 = plt.subplots(figsize=(20, 12))
    bp_r = ax1.boxplot(ranges_list[100:110], sym='+')
    fig.show()

    fig, ax1 = plt.subplots(figsize=(20, 12))
    bp_rr = ax1.boxplot(range_rates_list[100:110], sym='+')
    fig.show()

