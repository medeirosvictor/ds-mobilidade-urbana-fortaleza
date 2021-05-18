import os
from os import path
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import dateutil.parser
import numpy as np

final_df = pd.DataFrame()


# GET AVERAGES
#avg_per_hour = {key: (tuple((hour, []) for hour in range(24))) for key in list(data['linha'].unique())}

def generate_df(filename):
    print(filename)
    validation_georeffed_do_dia = pd.read_csv('./validation_proc/' + filename, sep=';')
    
    #print('Before Zero Drop:', len(validation_georeffed_do_dia))
    #validation_georeffed_do_dia = validation_georeffed_do_dia[validation_georeffed_do_dia.lat != 0.000000]
    #validation_georeffed_do_dia = validation_georeffed_do_dia[validation_georeffed_do_dia.lng != 0.000000]
    #print('After Zero Drop:', len(validation_georeffed_do_dia))

    validation_georeffed_do_dia['data_hora'] = pd.to_datetime(validation_georeffed_do_dia['data_hora'],
                                                              format='%Y-%m-%d %H:%M:%S', errors='coerce')
    validation_georeffed_do_dia['data_hora'] = validation_georeffed_do_dia['data_hora'].dt.floor('H')
    validation_georeffed_do_dia['linha'] = validation_georeffed_do_dia['linha'].astype(str)
    validation_georeffed_do_dia = pd.DataFrame(
        {'validations_per_hour': validation_georeffed_do_dia.groupby(['linha', 'data_hora']).size()}).reset_index()

    validation_georeffed_do_dia['dia_da_semana'] = validation_georeffed_do_dia.apply(lambda row: row.data_hora.dayofweek, axis=1)

    validation_georeffed_do_dia['hour_sin'] = np.sin(2 * np.pi * validation_georeffed_do_dia['data_hora'].dt.hour/23.0)
    validation_georeffed_do_dia['hour_cos'] = np.cos(2 * np.pi * validation_georeffed_do_dia['data_hora'].dt.hour/23.0)
    validation_georeffed_do_dia["hora"] = validation_georeffed_do_dia['data_hora'].dt.strftime('%H')
    validation_georeffed_do_dia["dia_do_mes"] = validation_georeffed_do_dia['data_hora'].dt.day
    validation_georeffed_do_dia["mes"] = validation_georeffed_do_dia['data_hora'].dt.month

    print(len(validation_georeffed_do_dia))
    global final_df
    final_df = pd.concat([final_df, validation_georeffed_do_dia], ignore_index=True)
    print(len(final_df))

if __name__ == '__main__':

    all_files = os.listdir('./validation_proc/')

    for filename in all_files:
        generate_df(filename)

    beforeGroup_df = final_df
    
    #14746591
    #5759
    #151954
    final_df.to_csv('./final_df.csv', sep=';', index=False)

