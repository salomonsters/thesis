import numpy as np
import pandas as pd
import geopandas
from nl_airspace_def import ehaa_airspace, hoofddorp_airspace
from nl_airspace_helpers import prepare_gdf_for_plotting, add_basemap
import matplotlib.pyplot as plt

airspace = ehaa_airspace[ehaa_airspace['airport'] == "EHRD"]
airspace = hoofddorp_airspace
show_only_track_in_airspace = False
aircraft_db = pd.read_csv('data/aircraft_db/aircraft_db.csv', dtype=str, index_col=0)

df = pd.read_csv('data/adsb_decoded/ADSB_DECODED_20180101.csv.gz')#, converters={'callsign': lambda s: s.replace('_', '')})
df_alt_min, df_alt_max = df['alt'].min(), df['alt'].max()

for fid, track in df.groupby('fid'):
    gdf_track = geopandas.GeoDataFrame(track, geometry=geopandas.points_from_xy(track.lon, track.lat))
    try:
        callsign = track.loc[track['callsign'].notnull()]['callsign'].unique()[0]
    except IndexError:
        callsign = "N/a"
    joined = geopandas.sjoin(gdf_track, airspace, how="inner", op="intersects")

    index_in_airspace = (joined.alt >= joined.lower_limit_ft) & (joined.alt <= joined.upper_limit_ft)
    if not np.any(index_in_airspace):
        print("Callsign {0} not overlapping with airspace".format(callsign))
        continue
    if show_only_track_in_airspace:
        gdf_track = joined[index_in_airspace]
    vmin = gdf_track.alt.min()
    vmax = gdf_track.alt.max()
    gdf_track_converted = prepare_gdf_for_plotting(gdf_track)
    fig = plt.figure()
    airspace_projected = prepare_gdf_for_plotting(airspace)
    ax = airspace_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    add_basemap(ax, zoom=9, ll=False)
    gdf_track_converted.plot(ax=ax, column='alt', cmap='plasma', legend=True, markersize=0.1, linewidth=1, vmin=vmin, vmax=vmax)
    plt.title(callsign)
    plt.show()
    if input("Continue?"):
        break


# gdf = geopandas.GeoDataFrame(df, geometry=geopandas.points_from_xy(df.lon, df.lat))
#
# ax = gdf.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
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
