import pandas as pd 
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Lê csv com base de dados
baseSalarios = pd.read_csv('salarios.csv')

# Retorna os 3 primeiros registros
print(baseSalarios.head(3))

# Guarda um array contendo todos os dados da coluna que será utilizada como previsora, 
# sendo obrigatório que seja uma coluna que tenha apenas 2 valores 
# (homem e mulher, no caso do exemplo) 
x = baseSalarios['sexo'].values

# Transforma todos os dados categóricos do array para numérico
label = LabelEncoder()
x = label.fit_transform(x)

# Retorna o formato atual do shape que estará em vetor
print(x.shape)

# Transforma o shape em uma matriz cologando uma coluna com valor de 1
x = x.reshape(-1, 1)

# Guarda um array com os dados que serão utilizados para previsão
y = baseSalarios['salario'].values

# Realiza o treinamento da base
classificador = BernoulliNB()
classificador.fit(x, y)

# Realiza as previsões e faz uma comparação delas com a informação da base
previsoes = classificador.predict(x)
print(previsoes, y)

# Verifica a precisão que o algoritmo teve comparando a base com as previsões
precisao = accuracy_score(y, previsoes)
print(precisao)

