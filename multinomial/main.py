import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Lê csv com base de dados
baseSalarios = pd.read_csv('salarios.csv')

# Retorna os 3 primeiros registros
print(baseSalarios.head(3))

# Transforma todos os dados categóricos do array para discretos
label_encoder0 = LabelEncoder()
baseSalarios['sexo'] = label_encoder0.fit_transform(baseSalarios['sexo'])

label_encoder1 = LabelEncoder() 
baseSalarios['graduacao'] = label_encoder1.fit_transform(baseSalarios['graduacao'])

label_encoder2 = LabelEncoder()
baseSalarios['idade'] = label_encoder2.fit_transform(baseSalarios['idade'])

# Retorna como ficou os registros após a transformação
print(baseSalarios.head(3))

# Cria a variavel que será utilizada para predição usando as 3 colunas de informações
x = baseSalarios.iloc[:, [1, 2, 3]].values
print(x)

# Guarda um array com os dados que serão utilizados para previsão
y = baseSalarios['salario'].values
print(y)

# Realiza o treinamento da base
classificador = MultinomialNB()
classificador.fit(x, y)

# Realiza as previsões e faz uma comparação delas com a informação da base
previsoes = classificador.predict(x)
print(previsoes, y)

# Verifica a precisão que o algoritmo teve comparando a base com as previsões
precisao = accuracy_score(y, previsoes)
print(precisao)
