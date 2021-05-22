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

st.markdown(
        """
    <style>
        .reportview-container .main .block-container{
            max-width: 1370px;
        }
    </style>
    """,
            unsafe_allow_html=True,
    )

# Reading Data file (geolocalized)
data = pd.read_csv('final_df.csv', sep=';')
data = data.drop_duplicates()

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
model_data = data[data.linha.isin([busline_filter])].reset_index()

X = model_data.drop(['data_hora','hora', 'mes', 'validations_per_hour'], axis='columns')
one_hot_encoder = OneHotEncoder(sparse=False)
X[['domingo','segunda', 'terca', 'quarta', 'quinta', 'sexta', 'sabado']] = one_hot_encoder.fit_transform(X['dia_da_semana'].values.reshape(-1,1))
y = model_data.validations_per_hour

LinearRegressionModel = LinearRegression()
RandomForestModel = RandomForestRegressor(n_jobs=6)

def get_chart_data(busline_filter):
    chart_data = model_data
    if busline_filter == 'Todos':
        chart_data = chart_data.drop(['mes', 'data_hora','dia_do_mes','hour_cos', 'hour_sin', 'dia_da_semana', 'linha'], axis='columns').set_index('hora')
        return chart_data
    chart_data = chart_data[chart_data.linha.isin([busline_filter])]
    chart_data = chart_data.drop(['mes', 'data_hora','dia_do_mes','hour_cos', 'hour_sin', 'dia_da_semana', 'linha'], axis='columns').set_index('hora')
    return chart_data

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

def single_busline_model(model):
    single_busline_model = model
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.1, random_state=5)
    single_busline_model.fit(X_train, Y_train)
    performance_scoring = get_performance(model, X_test, Y_test)
    return single_busline_model, performance_scoring

def train_28th_predict_29th(model):
    day29_model = model

    chosen_line_data = data[data.linha.isin([busline_filter])]

    day29_model_data = data[data.linha.isin([busline_filter])]
    day29_model_data = day29_model_data[(day29_model_data.dia_do_mes < 30) & (day29_model_data.mes == 11)]

    X = day29_model_data.drop(['data_hora','hora','dia_do_mes','mes', 'validations_per_hour'], axis='columns')
    y = day29_model_data.validations_per_hour
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
    day29_model.fit(X_train, Y_train)
    performance_scoring = get_performance(day29_model, X_test, Y_test)

    c_line_data = chosen_line_data[(chosen_line_data.dia_do_mes >= 30) & (chosen_line_data.mes == 11)]

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

def train_3week(model):
    week3_model = model

    chosen_line_data = data[data.linha.isin([busline_filter])]

    week3_model_data = data[data.linha.isin([busline_filter])]
    week3_model_data = week3_model_data[(week3_model_data.dia_do_mes < 24) & (week3_model_data.mes == 11)]

    X = week3_model_data.drop(['data_hora','hora','dia_do_mes','mes', 'validations_per_hour'], axis='columns')
    y = week3_model_data.validations_per_hour
    
    X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
    week3_model.fit(X_train, Y_train)

    c_line_data = chosen_line_data[(chosen_line_data.dia_do_mes >= 24) & (chosen_line_data.mes == 11)]

    shape = (len(c_line_data.dia_da_semana.unique()), len(c_line_data.hora.unique()))

    prev = np.zeros(shape)
    for idx, dia in enumerate(c_line_data.dia_da_semana.unique()):
        for index, hora  in enumerate(c_line_data.hora.unique()):
            h_sin= np.sin(2 * np.pi * hora/23.0)
            h_cos = np.cos(2 * np.pi * hora/23.0)
            prev[idx, index] = week3_model.predict([[busline_filter, idx, h_sin, h_cos]])
    
    for idx, dia in enumerate(c_line_data.dia_da_semana.unique()):
        actual = c_line_data[(chosen_line_data.dia_da_semana == idx)].validations_per_hour
        if len(actual) != len(prev[idx]):
            continue

        st.write('Dia da semana: <span class="text-light bg-dark p-2">', day_of_week_translator[idx], "</span>", unsafe_allow_html=True)
        fig = plt.figure(figsize=(12, 5))
        plt.xticks(c_line_data[(chosen_line_data.dia_da_semana == idx)].hora.unique())
        plt.plot(c_line_data[(chosen_line_data.dia_da_semana == idx)].hora.unique(), prev[idx], label="predicted")
        plt.plot(c_line_data[(chosen_line_data.dia_da_semana == idx)].hora.unique(), c_line_data[(chosen_line_data.dia_da_semana == idx)].validations_per_hour, label="actual")
        plt.legend(loc="upper right")
        st.pyplot(plt)
        st.write(pd.DataFrame({'predicted': prev[idx], 'actual': actual}))

    performance_scoring = get_performance(week3_model, X_test, Y_test)

    return week3_model, performance_scoring

# Main text setup
st.title('Analise Mobilidade Urbana - Fortaleza')

#Filters Setup
st.write('Quantidade de Linhas de Onibus', len(data.linha.unique()))

st.write('Quantidade média de validações por hora: ')

#st.bar_chart(get_chart_data(busline_filter), use_container_width=True)

st.markdown("""
<p class="alert alert-info">Existem linhas que nao seguem os padroes esperados de picos: <br>
    <strong>07:00 - 08:00</strong><br>
    <strong>12:00</strong><br>
    <strong>17:00 - 18:00</strong><br>
    Por exemplo: 055</p>
 """, unsafe_allow_html=True)

st.write("""
    ### Aplicando modelos para a linha de onibus <span class="text-light bg-dark p-2">"""
    + str(busline_filter)+"""</span>""", unsafe_allow_html=True)

features_col, target_col = st.beta_columns(2)
features_col.write("Features")
features_col.write(X)
target_col.write("Target")
target_col.write(y)

st.write("### Relacao Feature x Target")

st.write("###  1 Modelo Por Linha (mes completo de treino)")
model_per_line_col1, model_per_line_col2 = st.beta_columns(2)
model_per_line_col1.write('<span class="text-warning bg-dark p-2"> Regressao Linear </span>', unsafe_allow_html=True)
model_per_line_col2.write('<span class="text-warning bg-dark p-2">Random Forest</span>', unsafe_allow_html=True)


model_per_line_lr, model_per_line_lr_performance = single_busline_model(LinearRegressionModel)
model_per_line_rf, model_per_line_rf_performance = single_busline_model(RandomForestModel)

model_per_line_col1.write(model_per_line_lr_performance)
model_per_line_col2.write(model_per_line_rf_performance)

st.write("Utilizando sample aleatorio de dado para teste de previsao: ")

predict_test = X.sample(n=1)
predict_res = model_per_line_lr.predict(predict_test)
st.write(model_data[predict_test.index[0]:predict_test.index[0]+1])
st.write("Regressao Linear -> resultado do predict de test: ", predict_res)

predict_res2 = model_per_line_rf.predict(predict_test)
st.write("Random Forest -> resultado do predict de test: ", predict_res2)


st.write("###  Treina 28 dias -> predict dia 29")
day28_lr_col, day28_rf_col = st.beta_columns(2)
day28_lr_col.write('<span class="text-warning bg-dark p-2"> Regressao Linear </span>', unsafe_allow_html=True)
day28_rf_col.write('<span class="text-warning bg-dark p-2">Random Forest</span>', unsafe_allow_html=True)

with day28_lr_col:
    day28_lr_model, day28_lr_performance = train_28th_predict_29th(LinearRegressionModel)
    st.write("Performance: ", day28_lr_performance)
with day28_rf_col:
    day28_rf_model, day28_rf_performance =  train_28th_predict_29th(RandomForestModel)
    st.write("Performance: ", day28_rf_performance)

st.write("###  Treina 3 semanas -> predict semana 4")
week3_lr_col, week3_rf_col = st.beta_columns(2)
week3_lr_col.write('<span class="text-warning bg-dark p-2"> Regressao Linear </span>', unsafe_allow_html=True)
week3_rf_col.write('<span class="text-warning bg-dark p-2">Random Forest</span>', unsafe_allow_html=True)

with week3_lr_col:
    week3_lr_model, week3_lr_performance = train_3week(LinearRegressionModel)
    st.write("Performance: ", week3_lr_performance)
with week3_rf_col:
    week3_rf_model, week3_rf_performance = train_3week(RandomForestModel)
    st.write("Performance: ", week3_rf_performance)

#Processing the machine learning algorithm


### Styles needed for Dashboard yay
st.markdown(
    """
<link href="https://cdn.jsdelivr.net/npm/bootstrap@5.0.1/dist/css/bootstrap.min.css" rel="stylesheet" integrity="sha384-+0n0xVW2eSR5OomGNYDnhzAbDsOXxcvSN1TPprVMTNDbiYZCxYbOOl7+AMvyTG2x" crossorigin="anonymous">

""",
    unsafe_allow_html=True,
)

