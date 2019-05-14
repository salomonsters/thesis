import numpy as np
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt
import glob
import os

airspace = ehaa_airspace[ehaa_airspace['airport'] == "EHRD"]
show_only_track_in_airspace = False

## SINGLE FILENAME
# df = pd.read_csv('data/adsb_decoded_in_ehrd/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})
## ENTIRE DIRECTORY
path = './data/adsb_decoded_in_ehrd/'
df = pd.concat(map(pd.read_csv, glob.glob(os.path.join(path, "*.csv.gz"))))
df_alt_min, df_alt_max = df['alt'].min(), df['alt'].max()
fig = plt.figure()
airspace_projected = prepare_gdf_for_plotting(airspace)
ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
add_basemap(ax, zoom=9, ll=False)

gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
gdf_track_converted = prepare_gdf_for_plotting(gdf)
gdf_track_converted.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=0.1, vmin=df_alt_min, vmax=df_alt_max)
plt.show()
