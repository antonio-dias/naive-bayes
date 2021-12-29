# Naive Bayes

### Introdução

O algoritmo “*Naive Bayes*” é um classificador probabilístico muito utilizado em machine learning para categorizar textos com base na frequência das palavras usadas.

Ele recebe o nome de “*naive*” (ingênuo) porque desconsidera a correlação entre as variáveis (features). Ou seja, se determinada fruta é rotulada como “Limão”, caso ela também seja descrita como “Verde” e “Redonda”, o algoritmo não vai levar em consideração a correlação entre esses fatores. Isso porque trata cada um de forma independente. Por ter uma parte matemática relativamente simples, possui um bom desempenho e precisa de poucas observações para ter uma boa acurácia.

Entre as possibilidades de aplicações está a classificação de um e-mail como SPAM ou Não-SPAM, Análise de Sentimento nas redes sociais, análise de crédito, diagnósticos médicos e a identificação de um assunto com base em seu conteúdo.

### Matemática do algoritmo

O teorema de Bayes foi tirada da lei da probabilidade total, expresso matematicamente na forma da seguinte equação:

![](https://lh3.googleusercontent.com/pYoRBpN9RbLq665rngfxQ9Ph0CI_2CJ--BvByCmqYBTeiZF89cPbXYDb2x8AbXXUzvTyVON91KofbI3bxE8PZHSWPJddC_ABmdBw982ZWJikXHIYxaxpW9DxGWQrSOxURkKHYB4)

Onde:
- **P(A)** e **P(B)** são probabilidades a prioridade
- **P(A | B)** é a probabilidade de a prioridade **A** caso condicional **B** esteja positivo
- **P(B | A)** é a probabilidade de a prioridade **B** caso condicional **A** esteja positivo
    
Outra forma do teorema que é geralmente encontrada quando são consideradas duas afirmações ou hipóteses concorrentes:

![](https://lh3.googleusercontent.com/Fmuz86eJkPgAGeKBOoCTPG7TtjmF-VG_0VsedWR5Ux9fPoM1HJCA7x_ORS4ywpIGMtA9xyzzkNlfkNc7Ntov7P5Zqaah6EL3RCe1SkHG5yEt5vpnfEBqJCP8WWRPaDwxDaIEMnk)

Para proposição A e evidência B:
- **P(A)** é o grau de crença em **A**;
- **P(Aᶜ )** é a probabilidade correspondente do grau de crença inicial contra A. **P(Aᶜ ) = 1 - P(A)**;
- **P(B | A)** é o grau de crença em **B**, dado que a proposição **A** é verdadeira;
- **P(B | Aᶜ )** é o grau de crença em **B**, dado que a proposição **A** é falsa;
- **P(A | B)** é a probabilidade para **A**, após considerar **B** para e contra **A**.

Exemplo: 

Digamos que estamos trabalhando no diagnóstico de uma nova doença, e que fizemos testes em 100 pessoas distintas.
 
Após coletarmos a análise, descobrimos que 20 pessoas possuíam a doença (20%) e 80 pessoas estavam saudáveis (80%), sendo que das pessoas que possuíam a doença, 90% receberam Positivo no teste da doença, e 30% das pessoas que não possuíam a doença também receberam o teste positivo.

Listando esses dados de uma forma mais clara, temos:
- 100 pessoas realizaram o teste.
- 20% das pessoas que realizaram o teste possuíam a doença.
- 90% das pessoas que possuíam a doença, receberam positivo no teste.
- 30% das pessoas que não possuíam a doença, receberam positivo no teste.

A pergunta neste caso seria: Se uma nova pessoa realizar o teste e receber um resultado positivo, qual a probabilidade de ela possuir a doença?

O algoritmo de Naive Bayes consiste em encontrar uma probabilidade a posteriori (possuir a doença, dado que recebeu um resultado positivo), multiplicando a probabilidade a priori (possuir a doença) pela probabilidade de “receber um resultado positivo, dado que tem a doença”.

Devemos também computar a probabilidade a posteriori da negação (Não possuir a doença, dado que recebeu um resultado Positivo).

Ou seja:
- P(doença|positivo) = 20% * 90%
- P(doença|positivo) = 0,2 * 0,9
- P(doença|positivo) = 0,18
- P(não doença|positivo) = 80% * 30%
- P(não doença|positivo) = 0,8 * 0,3
- P(não doença|positivo) = 0,24
    
Após isso precisamos normalizar os dados, para que a soma das duas probabilidades resulte 1 (100%).

Para isso, dividimos o resultado pela soma das duas probabilidades.

Exemplo:
- P(doença|positivo) = 0,18/(0,18+0,24) = 0,4285
- P(não doença|positivo) = 0,24/(0,18+0,24) = 0,5714
- 0,4285 + 0,5714 = 0,9999.. ou aproximadamente 1.
    
Podemos concluir que se o resultado do teste da nova pessoa for positivo, ela possui aproximadamente 43% (0,4285) de chance de estar doente.

### Prós e Contras

Prós:
- É fácil e rápido para prever o conjunto de dados da classe de teste. Também tem um bom desempenho na previsão de classes múltiplas.
- Quando a suposição de independência prevalece, um classificador Naive Bayes tem melhor desempenho em comparação com outros modelos como regressão logística, e você precisa de menos dados de treinamento.
- O desempenho é bom em caso de variáveis categóricas de entrada comparada com a variáveis numéricas. Para variáveis numéricas, assume-se a distribuição normal (curva de sino, que é uma suposição forte).

Contras:
- Se a variável categórica tem uma categoria (no conjunto de dados de teste) que não foi observada no conjunto de dados de treinamento, então o modelo irá atribuir uma probabilidade de 0 (zero) e não será capaz de fazer uma previsão. Isso é muitas vezes conhecido como “Zero Frequency”. Para resolver isso, podemos usar a técnica de alisamento. Uma das técnicas mais simples de alisamento é a chamada estimativa de Laplace.
- Por outro lado, naive Bayes é também conhecido como um mau estimador, por isso, as probabilidades calculadas não devem ser levadas muito a sério.
- Outra limitação do Naive Bayes é a suposição de preditores independentes. Na vida real, é quase impossível ter um conjunto de indicadores que sejam completamente independentes.

### Implementação em Python

O Python possui grandes bibliotecas que tratam e implementam os algoritmos mais usados em Aprendizado de Máquina.

Sendo uma das mais famosas a biblioteca [Scikit Learn](https://scikit-learn.org/).

O Sklearn implementa o Naive Bayes de 3 formas:
1. ***Gaussian***;
	É usado na classificação e assume uma distribuição normal.

2. ***Multinomial***;  
    É usado para contagem discreta. Por exemplo, digamos que temos um problema de classificação de texto. Aqui podemos considerar tentativas de Bernoulli, que é um passo além e, em vez de “palavra que ocorre no documento”, temos “contar quantas vezes a palavra ocorre no documento”, você pode pensar nisso como “número de vezes que o número desfecho x_i é observado durante as n tentativas “.
   
3. ***Bernoulli***:  
	O modelo binomial é útil se os vetores são binários (ou seja, zeros e uns). Uma aplicação seria de classificação de texto com um modelo de ‘saco de palavras’ onde os 1s e 0s são “palavra ocorre no documento” e “palavra não ocorre no documento”, respectivamente.

### Código de exemplo

Abaixo um código de exemplo utilizando a forma normal de Gaussian:

Importa a biblioteca do modelo Naive Bayes Gaussiano 
```python
from sklearn.naive_bayes import GaussianNB
import numpy as np
```
Designa as variáveis previsor e alvo
```python
A = np.array([[-3,7],[1,5], [1,2], [-2,0], [2,3], [-4,0], [-1,1], [1,1], [-2,2], [2,7], [-4,1], [-2,7]])
B = np.array([3, 3, 3, 3, 4, 3, 3, 4, 3, 4, 4, 4])
```
Cria um classificador Gaussiano 
```python
model = GaussianNB()  
```
Treina o modelo usando os dados de treino 
```python
model.fit(A, B) 
```
Resultado de previsão 
```python
predicted= model.predict([[1,2],[3,4]])
print(predicted)
```
Saída de dados: **([3,4])**

### Projetos do repositório
- [Gaussian](https://github.com/antonio-dias/naive-bayes/tree/main/gaussian)
- [Bernoulli](https://github.com/antonio-dias/naive-bayes/tree/main/bernoulli)
- [Multinomial](https://github.com/antonio-dias/naive-bayes/tree/main/multinomial)