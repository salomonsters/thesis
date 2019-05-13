import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import geopandas
import contextily as ctx

def add_basemap(ax, zoom, url='http://tile.stamen.com/terrain/tileZ/tileX/tileY.png'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))

# matplotlib.rcParams['agg.path.chunksize'] = 10000

aircraft_db = pd.read_csv('data/aircraft_db/aircraft_db.csv', dtype=str, index_col=0)

# callsign_convertor = lambda s: s.replace('_', '')
df = pd.read_csv('data/adsb_decoded/ADSB_DECODED_20180102.csv.gz')#, converters={    'callsign': callsign_convertor})
gdf = geopandas.GeoDataFrame(
    df, geometry=geopandas.points_from_xy(df.lon, df.lat))

ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
add_basemap(ax, zoom=10)
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# i=0
# for callsign, track in adsb_decoded.groupby('callsign'):
#     if len(callsign) == 0:
#         continue
#     if 'ph-' in callsign:
#         # if i > 6:
#         #     break
#         i = i+1
#         lat = track['lat'].astype(float)
#         lon = track['lon'].astype(float)
#         alt = track['alt'].astype(float)
#
#         ax.scatter(lat, lon, alt, label=callsign)
# # plt.legend()
# plt.show()
#
