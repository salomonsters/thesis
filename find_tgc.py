import pandas as pd
import numpy as np
import os
import datetime

aircraft_db = pd.read_csv('data/aircraft_db/aircraft_db.csv', dtype=str, index_col=0).reset_index()
source_dir = 'data/adsb_decoded/2019/2019_05'

tgc_icao = aircraft_db.query("regid=='ph-jmp'")['icao'].iloc[0]


if __name__ == '__main__':
    matched_df_list=[]
    for filename in os.listdir(source_dir):
        if os.path.isfile(os.path.join(source_dir, filename)):
            print("{time}: Start pre-processing for {0}".format(filename, time=datetime.datetime.now()))
            df = pd.read_csv(os.path.join(source_dir, filename))
            matched_df_list.append(df.query("icao=='{0}'".format(tgc_icao)))
    df_matched = pd.concat(matched_df_list)
    df_matched.to_csv(os.path.join('data', 'tgc_tracks.csv'))
    print("{time}: Finished merging file {0}, saved to directory {1}".format('tgc_tracks.csv', 'data', time=datetime.datetime.now()))
