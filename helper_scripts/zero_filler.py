import pandas as pd
from calendar import monthrange
from math import ceil
import numpy as np
import datetime as dt

data = pd.read_csv('../data_input_nozerofill_2015.csv', sep=',', delimiter=',')
data_model = data.copy()

#usando apenas as top 100 linhas (linhas com mais exemplos)
top100_linhalist = data_model.groupby(data_model.linha).sum().reset_index().sort_values('validacoes_por_hora', ascending=False).index[:5].to_list()

def encode(data, col, max_val):
    data[col + '_sin'] = np.sin(2 * np.pi * data[col]/max_val)
    data[col + '_cos'] = np.cos(2 * np.pi * data[col]/max_val)
    return data
## Inserir zero rows
def week_of_month(dt):
    """ Returns the week of the month for the specified date.
    """

    first_day = dt.replace(day=1)

    dom = dt.day
    adjusted_dom = dom + first_day.weekday()

    return int(ceil(adjusted_dom/7.0))

# linha_list = data_model.linha.unique()

for linha in top100_linhalist:
    print("Linha: ", linha)
    currentDataModel_Linha = data_model.loc[data_model.linha == linha]
    for mes in range(1, 13):
        print(mes)
        currentDataModel_Mes = currentDataModel_Linha.loc[currentDataModel_Linha.mes == mes]
        dia_max = monthrange(2020, mes)[1] 
        for dia in range(1, dia_max+1):
            currentDataModel_Dia = currentDataModel_Mes.loc[currentDataModel_Mes.d_mes == dia]
            
            for hora in range(0, 24):
                currentDataModel_Hora = currentDataModel_Dia[currentDataModel_Dia.hora == hora]
                if currentDataModel_Hora.empty == False:
                    continue
                else:
                    # print('INSERINDO NO MES '+str(mes)+' NO DIA '+str(dia)+ ' NA HORA '+str(hora))
                    #2018-01-01 00:00:00
                    if (mes == 2 and dia > 28):
                        continue
                    dd = dt.datetime.strptime('2015-'+str(mes)+'-'+str(dia)+' '+str(hora)+':00:00', "%Y-%m-%d %H:00:00")

                    # validation_georeffed_do_dia['d_mes'] = encode(validation_georeffed_do_dia, 'd_mes', 31)
                    # validation_georeffed_do_dia['d_semana'] = encode(validation_georeffed_do_dia, 'd_semana', 6)
                    # validation_georeffed_do_dia['d_ano'] = encode(validation_georeffed_do_dia, 'd_ano', 366)
                    # validation_georeffed_do_dia['mes'] = encode(validation_georeffed_do_dia, 'mes', 12)
                    # validation_georeffed_do_dia['semana_do_mes'] = encode(validation_georeffed_do_dia, 'semana_do_mes', 4)
                    # validation_georeffed_do_dia['hora'] = encode(validation_georeffed_do_dia, 'hora', 23)

                    new_row = {
                            'linha': linha,
                            'data_hora': dd,
                            'validacoes_por_hora': 0,
                            'd_semana': dd.weekday(),
                            'd_mes': dia,
                            'd_ano': dd.timetuple().tm_yday,
                            'mes': mes,
                            'semana_do_mes': week_of_month(dd),
                            'hora': hora,
                            
                            'd_mes_sin': np.sin(2 * np.pi * dia/31),
                            'd_mes_cos': np.cos(2 * np.pi * dia/31),

                            'd_semana_sin': np.sin(2 * np.pi *  dd.weekday()/7),
                            'd_semana_cos': np.cos(2 * np.pi *  dd.weekday()/7),

                            'd_ano_sin': np.sin(2 * np.pi * dd.timetuple().tm_yday/366),
                            'd_ano_cos': np.cos(2 * np.pi * dd.timetuple().tm_yday/366),

                            'mes_sin': np.sin(2 * np.pi * mes/12),
                            'mes_cos': np.cos(2 * np.pi * mes/12),
                            
                            'semana_do_mes_sin': np.sin(2 * np.pi * week_of_month(dd)/4),
                            'semana_do_mes_cos': np.cos(2 * np.pi * week_of_month(dd)/4),

                            'hora_sin': np.sin(2 * np.pi * hora/23),
                            'hora_cos': np.cos(2 * np.pi * hora/23)
                        }

                    data_model = data_model.append(new_row, ignore_index=True)

data_model = data_model.sort_values(['linha', 'data_hora'], ascending=[True, True])

data_model.to_csv('../data_input_zerofill_2015_top10_ciclycal.csv', index=False, sep=';')