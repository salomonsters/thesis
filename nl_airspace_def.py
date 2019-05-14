import geopandas
import pandas as pd

from nl_airspace_helpers import create_ctr, parse_tma, parse_lat_lon, add_basemap, prepare_gdf_for_plotting

### AMSTERDAM
# TMA strings: Taken directly from the AIP
eham_tma_strings = ["""Schiphol terminal control area 1

52°41'36.06"N 004°16'15.33"E;
52°48'20.00"N 005°20'00.00"E;
52°25'45.00"N 005°40'52.00"E;
52°22'41.00"N 005°40'05.00"E;
52°18'11.45"N 005°36'33.78"E;
52°11'45.00"N 005°04'24.00"E;
51°53'11.00"N 004°49'40.72"E;
51°56'10.00"N 004°21'15.00"E;
51°59'20.00"N 004°06'40.00"E;
52°17'06.01"N 003°59'10.51"E;
to point of origin.

When active, ATZ Lelystad B, glider areas Castricum 2, Hoek van Holland, and Valkenburg are excluded.

FL 095
1500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 2

52°54'00.65"N 004°07'04.77"E; 
52°48'19.15"N 004°21'00.00"E; 
52°17'06.01"N 003°59'10.51"E; 
52°17'29.93"N 003°41'47.07"E; 
to point of origin.

FL 055
3500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 3

52°11'45.00"N 005°04'24.00"E; 
52°12'18.66"N 005°07'09.97"E; 
51°53'11.00"N 005°05'47.00"E; 
along parallel to 
51°53'11.00"N 004°49'40.72"E; 
to point of origin.

FL 095
2500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 4

52°12'18.66"N 005°07'09.97"E; 
52°15'55.90"N 005°25'10.86"E; 
52°02'51.89"N 005°06'28.85"E; 
to point of origin.

FL 095
3500 ft AMSL

Class of airspace: A""",
               """Schiphol terminal control area 5

52°15'55.90"N 005°25'10.86"E;
52°18'11.45"N 005°36'33.78"E;
52°01'52.66"N 005°23'53.44"E;
51°53'11.00"N 005°05'47.00"E;
52°02'51.89"N 005°06'28.85"E;
to point of origin.

FL 095
FL 055

Class of airspace: A""", """Schiphol terminal control area 6

52°48'19.15"N 004°21'00.00"E; 
52°45'25.00"N 004°28'03.00"E; 
52°43'30.00"N 004°33'40.00"E; 
52°41'36.06"N 004°16'15.33"E; 
to point of origin.

FL 095
3500 ft AMSL

Class of airspace: A"""]

eham_ctr_centre = """52°18'29.00"N 004°45'51.00"E"""
eham_ctr_radius_in_nm = 8

#CTR string: remove the arc-component-lines and implement them separately
eham_ctr1_string = """52°24'54.39"N 004°38'05.45"E; 
52°28'21.18"N 004°38'24.39"E; 
52°28'01.08"N 004°48'12.13"E; 
52°26'21.48"N 004°48'03.00"E;
to point of origin."""

#CTR string: remove the arc-component-lines and implement them separately
eham_ctr2_string = """52°34'03.30"N 004°36'30.80"E; 
52°34'03.51"N 004°48'45.45"E; 
52°28'01.08"N 004°48'12.13"E; 
52°28'21.18"N 004°38'24.39"E; 
52°24'54.39"N 004°38'05.45"E; 
52°18'55.05"N 004°32'50.08"E; 
52°25'50.49"N 004°33'28.09"E; 
52°29'05.83"N 004°34'39.04"E; 
to point of origin."""

#CTR string: remove the arc-component-lines and implement them separately
eham_ctr3_string = """52°16'01.34"N 004°33'27.36"E; 
along anti-clockwise arc (radius 8 NM, centre 52°18'29.00"N 004°45'51.00"E) to 
52°10'46.46"N 004°49'15.93"E; 
52°04'02.84"N 004°48'38.87"E; 
along parallel to 
52°04'02.84"N 004°33'27.36"E; 
to point of origin."""

### ROTTERDAM
# TMA strings: Taken directly from the AIP
ehrd_tma_strings = [
    """Rotterdam terminal control area 1

52°17'06.01"N 003°59'10.51"E;
51°59'20.00"N 004°06'40.00"E;
51°56'10.00"N 004°21'15.00"E;
51°53'11.00"N 004°49'40.72"E;
51°36'00.00"N 004°36'15.29"E;
along parallel to 
51°36'00.00"N 004°11'32.79"E;
51°37'01.21"N 004°10'42.31"E;
51°36'47.89"N 004°07'33.20"E;
51°38'00.00"N 004°05'54.80"E;
along parallel to 
51°38'00.00"N 004°04'19.30"E;
51°36'54.29"N 003°59'09.57"E;
51°37'45.27"N 003°54'51.73"E;
51°56'25.84"N 003°45'01.84"E;
to point of origin.

FL 055
1500 ft AMSL

Class of airspace: E""",
    """Rotterdam terminal control area 2

51°56'25.84"N 003°45'01.84"E;
51°37'45.27"N 003°54'51.73"E;
51°35'44.20"N 003°52'07.46"E;
51°36'16.52"N 003°50'18.27"E;
51°36'18.84"N 003°47'57.34"E;
51°35'49.85"N 003°44'48.16"E;
51°35'58.94"N 003°40'52.93"E;
51°35'06.86"N 003°37'37.99"E;
51°35'38.76"N 003°35'00.00"E;
51°35'50.00"N 003°31'10.14"E;
to point of origin.

FL 055
2500 ft AMSL

Class of airspace: E""",
    """Rotterdam terminal control area 3

52°17'29.93"N 003°41'47.07"E;
52°17'06.01"N 003°59'10.51"E;
51°35'50.00"N 003°31'10.14"E;
along parallel to 
51°35'50.00"N 003°13'49.65"E;
to point of origin.

FL 055
3500 ft AMSL

Class of airspace: E"""
]

ehrd_ctr_centre = """51°57'25.00"N 004°26'14.00"E"""
ehrd_ctr_radius_in_nm = 8

#CTR string: remove the arc-component-lines and implement them separately
ehrd_ctr_string = """51°50'51.97"N 004°18'50.32"E; 
51°48'21.82"N 004°12'39.13"E; 
51°53'22.62"N 004°07'21.16"E; 
51°55'53.02"N 004°13'32.74"E; 
to point of origin."""

eham_ctr_1 = create_ctr("Schiphol CTR1", "C", 0, 3000,
                        centre=parse_lat_lon(eham_ctr_centre), radius_in_nm=eham_ctr_radius_in_nm,
                        extra_points=eham_ctr1_string, set_operation="union")
eham_ctr = [eham_ctr_1,
            create_ctr("Schiphol CTR2", "C", 1200, 3000, extra_points=eham_ctr2_string, geometry=eham_ctr_1['geometry'],
                       set_operation="difference"),
            create_ctr("Schiphol CTR3", "C", 1200, 3000, extra_points=eham_ctr3_string, geometry=eham_ctr_1['geometry'],
                       set_operation="difference")
            ]
eham_tma = [parse_tma(tma_s) for tma_s in eham_tma_strings]
eham_airspace = pd.DataFrame.from_records(eham_tma + eham_ctr)

ehrd_tma = [parse_tma(tma_s) for tma_s in ehrd_tma_strings]
ehrd_ctr = [create_ctr("Rotterdam CTR", "C", 0, 3000,
                       centre=parse_lat_lon(ehrd_ctr_centre), radius_in_nm=ehrd_ctr_radius_in_nm,
                       extra_points=ehrd_ctr_string, set_operation="union")]
ehrd_airspace = pd.DataFrame.from_records(ehrd_tma + ehrd_ctr)

ehaa_airspace = geopandas.GeoDataFrame(ehrd_airspace).append(geopandas.GeoDataFrame(eham_airspace))
ehaa_airspace.crs = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"

if __name__ == "__main__":
    import matplotlib.pyplot as plt

    plt.figure()
    gdf_projected = prepare_gdf_for_plotting(ehaa_airspace)
    plt.figure()
    ax = gdf_projected.plot(figsize=(10, 10), alpha=0.5, edgecolor='k')
    add_basemap(ax, zoom=9, ll=False)
    plt.title("Projected")
    plt.show()
