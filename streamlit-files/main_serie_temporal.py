import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
import streamlit as st
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from math import sqrt


st.markdown(
    """
    <style>
        .reportview-container .main .block-container{
            max-width: 1250px;
        }
    </style>
    
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-+0n0xVW2eSR5OomGNYDnhzAbDsOXxcvSN1TPprVMTNDbiYZCxYbOOl7+AMvyTG2x" crossorigin="anonymous">

    """,
    unsafe_allow_html=True,
)

st.title('Analise Mobilidade Urbana - Fortaleza')
st.subheader('Serie Temporal')

data = pd.read_csv('df_input.csv', sep=';')

day_of_week_translator = {
    0: "Domingo",
    1: "Segunda",
    2: "Terca",
    3: "Quarta",
    4: "Quinta",
    5: "Sexta",
    6: "Sabado",
}

model_options = ['Regressao linear', 'Random Forest']
busline_filter = st.sidebar.selectbox('Selecionar Linha:', list(data['linha'].unique()))
# data_model = data[data.linha.isin([busline_filter])]
data_model = data
st.write(data_model)

plt.figure(figsize=(15,5))
plt.xlabel('Tempo')
plt.ylabel('Quantidade')
plt.plot(data['d_semana'], data['validations_per_hour'], 'b--')
st.pyplot(plt)
plt.clf()
plt.plot(data['d_ano'], data['validations_per_hour'], 'b--')
st.pyplot(plt)


time_steps = 30  #TAMANHO DA JANELA
test_size = 120  #HORIZONTE DE PREVISÃO
one_hot_encoder = OneHotEncoder(sparse=False)

data_model[['domingo','segunda', 'terca', 'quarta', 'quinta', 'sexta', 'sabado']] = one_hot_encoder.fit_transform(data_model['d_semana'].values.reshape(-1,1))
data_model[['marco', 'abril', 'maio', 'junho', 'julho', 'agosto', 'setembro', 'outubro', 'novembro', 'dezembro', 'janeiro']] = one_hot_encoder.fit_transform(data_model['mes'].values.reshape(-1,1))

train_size = int(len(data_model)-(test_size))
train, test = data_model.iloc[0:train_size], data_model.iloc[(train_size-time_steps):len(data_model)]
st.write(len(train), len(test))

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps),0].to_numpy()
        v = np.append(v,X.iloc[i + time_steps,0])#linha
        v = np.append(v,X.iloc[i + time_steps,3])#d_semana
        # v = np.append(v,X.iloc[i + time_steps,10])#d_semanaonehotencoded -> provou ser PIOR
        # v = np.append(v,X.iloc[i + time_steps,11])#d_semanaonehotencoded -> provou ser PIOR
        # v = np.append(v,X.iloc[i + time_steps,12])#d_semanaonehotencoded -> provou ser PIOR
        # v = np.append(v,X.iloc[i + time_steps,13])#d_semanaonehotencoded -> provou ser PIOR
        # v = np.append(v,X.iloc[i + time_steps,14])#d_semanaonehotencoded -> provou ser PIOR
        # v = np.append(v,X.iloc[i + time_steps,15])#d_semanaonehotencoded -> provou ser PIOR
        # v = np.append(v,X.iloc[i + time_steps,16])#d_semanaonehotencoded -> provou ser PIOR
        v = np.append(v,X.iloc[i + time_steps,4])#hr_sin
        v = np.append(v,X.iloc[i + time_steps,5])#hr_cos
        v = np.append(v,X.iloc[i + time_steps,7])#d_mes -> provou ser melhor com a adicao deste componente
        # v = np.append(v,X.iloc[i + time_steps,8])#d_ano
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)
st.write(train)
st.write(test)
X_train, y_train = create_dataset(train, train['validations_per_hour'], time_steps)
X_test, y_test = create_dataset(test, test['validations_per_hour'], time_steps)

st.write(len(X_train), len(X_test))

model = RandomForestRegressor(n_jobs=10).fit(X_train,y_train)
preds = []

base_teste = np.copy(X_test)

for i in range(len(base_teste)):
    y_pred = model.predict(np.array([base_teste[i]]))[0]

    preds.append(y_pred)
    
    for k in range(len(preds)):
        
        if i<len(base_teste):
            if k < time_steps:
                if(i<len(base_teste)-1):
                    base_teste[i+1][(time_steps-1)-k] = preds[(len(preds)-1)-k]

st.write(len(preds))

dados_real = data_model.iloc[(train_size):len(data_model),2].to_numpy()

len(dados_real)

df_real_predito = pd.DataFrame({'real':dados_real,'predito':preds})

st.write(df_real_predito)
# X_train[0]

plt.figure(figsize=(15,5))
# plt.plot(range(len(y_train)),y_train, 'g--')
plt.plot(range(len(df_real_predito['predito'])),df_real_predito['predito'], 'g--')
plt.plot(range(len(df_real_predito['real'])),df_real_predito['real'], 'b')
# plt.xlim(0,30)
st.write('rmse=',sqrt(mean_squared_error(df_real_predito['real'].array,df_real_predito['predito'].array)))
st.write('mae=',mean_absolute_error(df_real_predito['real'].array,df_real_predito['predito'].array))
st.write('mape=',mean_absolute_percentage_error(df_real_predito['real'].array,df_real_predito['predito'].array))
st.write('r2=',r2_score(df_real_predito['real'].array,df_real_predito['predito'].array))

st.pyplot(plt)

def getModel(model_filter):
    if model_filter == 'Regressao Linear':
        md = LinearRegression()
    elif model_filter == 'Random Forest':
        md = RandomForestRegressor(n_jobs=6)
    return md

def get_performance(model, X_test, Y_test):
    y_test_predict = model.predict(X_test)
    mse = mean_squared_error(Y_test, y_test_predict)
    rmse = (np.sqrt(mse))
    r2 = r2_score(Y_test, y_test_predict)
    mean = mean_absolute_error(Y_test, y_test_predict)
    mape = mean_absolute_percentage_error(Y_test, y_test_predict)
    performance_scoring = [
        ("MSE", mse),
        ("RMSE", rmse),
        ("R2", r2),
        ("MAE", mean),
        ("MAPE", mape)
    ]
    performance_scoring = pd.DataFrame(performance_scoring,columns=['Metrica', 'Score'])
    performance_scoring['Score'] = performance_scoring['Score'].astype('float64')
    return performance_scoring

def single_busline_model(model_type):
    model = getModel(model_type)
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state=5)
    model.fit(X_train, Y_train)
    performance_scoring = get_performance(model, X_test, Y_test)
    return model, performance_scoring

def train_28th_predict_29th(model_type):
    day29_model = getModel(model_type)

    chosen_line_data = data[data.linha.isin([busline_filter])]

    day29_data_model = data[data.linha.isin([busline_filter])]
    day29_data_model = day29_data_model[(day29_data_model.d_mes < 30) & (day29_data_model.mes == 11)]

    X = day29_data_model.drop(['data_hora','hora','d_mes','mes', 'validations_per_hour'], axis='columns')
    y = day29_data_model.validations_per_hour
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
    day29_model.fit(X_train, Y_train)
    performance_scoring = get_performance(day29_model, X_test, Y_test)

    c_line_data = chosen_line_data[(chosen_line_data.d_mes >= 30) & (chosen_line_data.mes == 11)]

    prev = np.zeros(len(c_line_data.hora.unique()))
    for index, hora  in enumerate(c_line_data.hora.unique()):
        h_sin= np.sin(2 * np.pi * hora/23.0)
        h_cos = np.cos(2 * np.pi * hora/23.0)
        prev[index] = day29_model.predict([[busline_filter, 1, h_sin, h_cos]])

    
    actual = c_line_data.validations_per_hour

    

    fig = plt.figure(figsize=(12, 5))
    plt.xticks(c_line_data.hora.unique())
    plt.plot(c_line_data.hora.unique(), prev, label="predicted")
    plt.plot(c_line_data.hora.unique(), c_line_data.validations_per_hour, label="actual")
    plt.legend(loc="upper right")
    st.pyplot(plt)
    st.write(pd.DataFrame({'predicted': prev, 'actual': actual}))

    return day29_model, performance_scoring

def get_chart_data(busline_filter):
    chart_data = data
    if busline_filter == 'Todos':
        chart_data = chart_data.drop(['mes', 'data_hora','d_mes','hour_cos', 'hour_sin', 'd_semana', 'linha'], axis='columns').set_index('hora')
        return chart_data
    chart_data = chart_data[chart_data.linha.isin([busline_filter])]
    chart_data = chart_data.drop(['mes', 'data_hora','d_mes','hour_cos', 'hour_sin', 'd_semana', 'linha'], axis='columns').set_index('hora')
    return chart_data

def train_3week(model_type):
    week3_model = getModel(model_type)

    chosen_line_data = data[data.linha.isin([busline_filter])]

    week3_data_model = data[data.linha.isin([busline_filter])]
    week3_data_model = week3_data_model[(week3_data_model.d_mes < 24) & (week3_data_model.mes == 11)]

    X = week3_data_model.drop(['data_hora','hora','d_mes','mes', 'validations_per_hour'], axis='columns')
    y = week3_data_model.validations_per_hour
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
    week3_model.fit(X_train, Y_train)

    c_line_data = chosen_line_data[(chosen_line_data.d_mes >= 24) & (chosen_line_data.mes == 11)]

    shape = (len(c_line_data.d_semana.unique()), len(c_line_data.hora.unique()))

    prev = np.zeros(shape)
    for idx, dia in enumerate(c_line_data.d_semana.unique()):
        for index, hora  in enumerate(c_line_data.hora.unique()):
            h_sin= np.sin(2 * np.pi * hora/23.0)
            h_cos = np.cos(2 * np.pi * hora/23.0)
            prev[idx, index] = week3_model.predict([[busline_filter, idx, h_sin, h_cos]])
    
    for idx, dia in enumerate(c_line_data.d_semana.unique()):
        actual = c_line_data[(chosen_line_data.d_semana == idx)].validations_per_hour
        if len(actual) != len(prev[idx]):
            continue

        st.write('Dia da semana: <span class="text-light bg-dark p-2">', day_of_week_translator[idx], "</span>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(12, 5))
        plt.xticks(c_line_data[(chosen_line_data.d_semana == idx)].hora.unique())
        plt.plot(c_line_data[(chosen_line_data.d_semana == idx)].hora.unique(), prev[idx], label="predicted")
        plt.plot(c_line_data[(chosen_line_data.d_semana == idx)].hora.unique(), c_line_data[(chosen_line_data.d_semana == idx)].validations_per_hour, label="actual")
        plt.legend(loc="upper right")
        st.pyplot(plt)
        st.write(pd.DataFrame({'predicted': prev[idx], 'actual': actual}))

    performance_scoring = get_performance(week3_model, X_test, Y_test)

    return week3_model, performance_scoring


# #Filters Setup
# st.write('Quantidade de Linhas de Onibus', len(data.linha.unique()))

# st.write('Quantidade média de validações por hora: ')

# #st.bar_chart(get_chart_data(busline_filter), use_container_width=True)

# st.markdown("""
# <p class="alert alert-info">Existem linhas que nao seguem os padroes esperados de picos: <br>
#     <strong>07:00 - 08:00</strong><br>
#     <strong>12:00</strong><br>
#     <strong>17:00 - 18:00</strong><br>
#     Por exemplo: 055</p>
#  """, unsafe_allow_html=True)

# st.write("""
#     ### Aplicando modelos para a linha de onibus <span class="text-light bg-dark p-2">"""
#     + str(busline_filter)+"""</span>""", unsafe_allow_html=True)

# features_col, target_col = st.beta_columns(2)
# features_col.write("Features")
# features_col.write(X)
# target_col.write("Target")
# target_col.write(y)


# st.write("###  1 Modelo Por Linha (mes completo de treino)")
# model_per_line_col1, model_per_line_col2 = st.beta_columns(2)
# model_per_line_col1.write('<span class="text-warning bg-dark p-2"> Regressao Linear </span>', unsafe_allow_html=True)
# model_per_line_col2.write('<span class="text-warning bg-dark p-2">Random Forest</span>', unsafe_allow_html=True)


# model_per_line_lr, model_per_line_lr_performance = single_busline_model('Regressao Linear')
# model_per_line_rf, model_per_line_rf_performance = single_busline_model('Random Forest')

# model_per_line_col1.write(model_per_line_lr_performance)
# model_per_line_col2.write(model_per_line_rf_performance)

# st.write("Utilizando sample aleatorio de dado para teste de previsao: ")

# predict_test = X.sample(n=1)
# st.write(data[predict_test.index[0]:predict_test.index[0]+1])

# predict_res = model_per_line_lr.predict(predict_test)
# st.write("Regressao Linear -> resultado do predict de test: ", predict_res)

# predict_res2 = model_per_line_rf.predict(predict_test)
# st.write("Random Forest -> resultado do predict de test: ", predict_res2)


# st.write("###  Treina 28 dias -> predict dia 29")
# day28_lr_col, day28_rf_col = st.beta_columns(2)
# day28_lr_col.write('<span class="text-warning bg-dark p-2"> Regressao Linear </span>', unsafe_allow_html=True)
# day28_rf_col.write('<span class="text-warning bg-dark p-2">Random Forest</span>', unsafe_allow_html=True)

# with day28_lr_col:
#     day28_lr_model, day28_lr_performance = train_28th_predict_29th('Regressao Linear')
#     st.write("Performance: ", day28_lr_performance)
# with day28_rf_col:
#     day28_rf_model, day28_rf_performance =  train_28th_predict_29th('Random Forest')
#     st.write("Performance: ", day28_rf_performance)

# st.write("###  Treina 3 semanas -> predict semana 4")
# week3_lr_col, week3_rf_col = st.beta_columns(2)
# week3_lr_col.write('<span class="text-warning bg-dark p-2"> Regressao Linear </span>', unsafe_allow_html=True)
# week3_rf_col.write('<span class="text-warning bg-dark p-2">Random Forest</span>', unsafe_allow_html=True)

# with week3_lr_col:
#     week3_lr_model, week3_lr_performance = train_3week('Regressao Linear')
#     st.write("Performance: ", week3_lr_performance)
# with week3_rf_col:
#     week3_rf_model, week3_rf_performance = train_3week('Random Forest')
#     st.write("Performance: ", week3_rf_performance)

