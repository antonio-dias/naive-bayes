import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

# Lê csv com base de dados
irisDB = pd.read_csv("Iris.csv")

# Retorna o tamanho da base (linhas, colunas)
print(irisDB.shape)

# Retorna os 3 primeiros registros
print(irisDB.head(3))

# Transforma os nomes simbólicos das espécies em números adequados para o classificador de Bayes
label = LabelEncoder()
label.fit(irisDB['Species'])
irisDB['Species'] = label.transform(irisDB['Species'])

# Divide o conjunto de dados em 2/3 de dados de treinamento e 1/3 de dados de teste
treinamentoSet, testSet = train_test_split(irisDB, test_size = 0.33)

# Retorna o tamanho da base de treinamento (linhas, colunas)
print(treinamentoSet.shape)

# Retorna o tamanho da base de teste (linhas, colunas)
print(testSet.shape)

# Retorna os 3 primeiros registros da base de treinamento
print(treinamentoSet.head(3))

# Formata os dados e valores esperados para SKLearn - Treinamento
treinamentoData = pd.DataFrame.to_numpy(treinamentoSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
treinamentoTarget = pd.DataFrame.to_numpy(treinamentoSet[['Species']]).ravel()

# Formata os dados e valores esperados para SKLearn - Teste
testeData = pd.DataFrame.to_numpy(testSet[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']])
testeTarget = pd.DataFrame.to_numpy(testSet[['Species']]).ravel()

classificador = GaussianNB()
classificador.fit(treinamentoData, treinamentoTarget)

valoresPrevistos = classificador.predict(testeData)

nErrors = (testeTarget != valoresPrevistos).sum()
precisao = 1.0 - nErrors / testeTarget.shape[0]
print("Precisao: ", precisao)