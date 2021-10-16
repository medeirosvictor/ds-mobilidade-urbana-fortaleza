import os
from os import path
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import dateutil.parser
import numpy as np
import datetime
import calendar

final_df = pd.DataFrame()

def fill_header(val_do_dia, filename):
    # 777;84;Siqueira/Messejana/P;2001;06/01/2015 09:21:03;2;02-ESTUDANTE ETUFOR;Ida;N;
    val_do_dia.to_csv('dfs_2015/' + filename, header=['id_onibus','linha','linha_nome','undf','data_hora','undf2','tipo_validacao','sentido','integracao'], index=False)
    df = pd.read_csv('dfs_2015/' + filename)
    return df


def week_of_month(tgtdate):
    tgtdate = tgtdate.to_pydatetime()

    days_this_month = calendar.mdays[tgtdate.month]
    for i in range(1, days_this_month):
        d = datetime.datetime(tgtdate.year, tgtdate.month, i)
        if d.day - d.weekday() > 0:
            startdate = d
            break
    # now we canuse the modulo 7 appraoch
    return (tgtdate - startdate).days //7 + 1

def generate_df(filename):
    print(filename)
    val = pd.read_csv('../2015/' + filename, sep=';')
    val = val.drop('Unnamed: 9', axis=1)
    val.reset_index(drop=True, inplace=True)
    # print(val.info())
    validation_georeffed_do_dia = fill_header(val, filename)
    # validation_georeffed_do_dia.info()
    
    #print('Before Zero Drop:', len(validation_georeffed_do_dia))
    #validation_georeffed_do_dia = validation_georeffed_do_dia[validation_georeffed_do_dia.lat != 0.000000]
    #validation_georeffed_do_dia = validation_georeffed_do_dia[validation_georeffed_do_dia.lng != 0.000000]
    #print('After Zero Drop:', len(validation_georeffed_do_dia))
    # validation_georeffed_do_dia.info()
    # id_onibus,linha,linha_nome,ano_onibus,data_hora,alg,tipo_validacao,sentido,integracao
    validation_georeffed_do_dia['id_onibus'] = validation_georeffed_do_dia['id_onibus'].astype(np.int32)
    validation_georeffed_do_dia['linha'] = validation_georeffed_do_dia['linha'].astype(str)
    validation_georeffed_do_dia['linha_nome'] = validation_georeffed_do_dia['linha_nome'].astype(str)
    validation_georeffed_do_dia['undf'] = validation_georeffed_do_dia['undf'].astype(str)
    validation_georeffed_do_dia['data_hora'] = pd.to_datetime(validation_georeffed_do_dia['data_hora'],
                                                              format='%d/%m/%Y %H:%M:%S', errors='coerce')
    validation_georeffed_do_dia['data_hora'] = validation_georeffed_do_dia['data_hora'].dt.floor('H')
    # validation_georeffed_do_dia['undf2'] = validation_georeffed_do_dia['undf2'].astype(np.int32)s
    validation_georeffed_do_dia['tipo_validacao'] = validation_georeffed_do_dia['tipo_validacao'].astype(str)
    validation_georeffed_do_dia['sentido'] = validation_georeffed_do_dia['sentido'].astype(str)
    validation_georeffed_do_dia['integracao'] = validation_georeffed_do_dia['integracao'].astype(str)
    

    validation_georeffed_do_dia = pd.DataFrame(
        {'validations_per_hour': validation_georeffed_do_dia.groupby(['linha', 'data_hora']).size()}).reset_index()

    validation_georeffed_do_dia['d_semana'] = validation_georeffed_do_dia.apply(lambda row: row.data_hora.dayofweek, axis=1)

    validation_georeffed_do_dia['hour_sin'] = np.sin(2 * np.pi * validation_georeffed_do_dia['data_hora'].dt.hour/23.0)
    validation_georeffed_do_dia['hour_cos'] = np.cos(2 * np.pi * validation_georeffed_do_dia['data_hora'].dt.hour/23.0)
    validation_georeffed_do_dia["hora"] = validation_georeffed_do_dia['data_hora'].dt.strftime('%H')
    validation_georeffed_do_dia["d_mes"] = validation_georeffed_do_dia['data_hora'].dt.day
    validation_georeffed_do_dia["d_ano"] = validation_georeffed_do_dia['data_hora'].dt.dayofyear
    validation_georeffed_do_dia["mes"] = validation_georeffed_do_dia['data_hora'].dt.month
    validation_georeffed_do_dia["semana_do_mes"] = validation_georeffed_do_dia['data_hora'].apply(week_of_month)

    print(len(validation_georeffed_do_dia))
    global final_df
    final_df = pd.concat([final_df, validation_georeffed_do_dia], ignore_index=True)
    print(len(final_df))

if __name__ == '__main__':

    all_files = os.listdir('../2015/')

    for filename in all_files:
        generate_df(filename)

    final_df.to_csv('../data_input_nozerofill_2015.csv', sep=',', index=False)
