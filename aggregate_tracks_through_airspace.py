import numpy as np
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace
import datetime
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool, freeze_support


airspace = ehaa_airspace[ehaa_airspace['airport'] == "EHRD"]
source_dir = 'data/adsb_decoded/'
export_dir = 'data/adsb_decoded_in_ehrd/'


def return_gdf_if_in_airspace(iter):
    fid, track = iter
    gdf_track = geopandas.GeoDataFrame(track, geometry=geopandas.points_from_xy(track.lon, track.lat))
    gdf_track.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
    joined = geopandas.sjoin(gdf_track, airspace, how="inner", op="intersects")
    # Join only works in 2d, need to check if the joined segments match the corresponding altitude limits
    index_in_airspace = (joined.alt >= joined.lower_limit_ft) & (joined.alt <= joined.upper_limit_ft)
    if np.any(index_in_airspace):
        return track
    else:
        return None


if __name__ == '__main__':
    freeze_support()
    pool = Pool(processes=11)  # start this many worker processes
    if not os.path.isdir(export_dir):
        raise FileNotFoundError("ERROR: Export directory {0} not found, aborting.".format(export_dir))
    for filename in os.listdir(source_dir):
        filename_export = os.path.join(export_dir, filename)
        if os.path.exists(filename_export):
            print("{time}: Export file {0} already exists; skipping".format(filename, time=datetime.datetime.now()))
            continue
        print("{time}: Start pre-processing for {0}".format(filename, time=datetime.datetime.now()))
        df = pd.read_csv(os.path.join(source_dir, filename))
        columns = list(df.columns)
        matched_df_list = pool.map(return_gdf_if_in_airspace, df.groupby('fid'))
        print("{time}: Finished pre-processing for {0}, starting merge".format(filename, time=datetime.datetime.now()))
        df_matched = pd.concat(matched_df_list)
        df_matched.to_csv(os.path.join(export_dir, filename), columns=columns)
        print("{time}: Finished merging file {0}, saved to directory {1}".format(filename, export_dir, time=datetime.datetime.now()))



