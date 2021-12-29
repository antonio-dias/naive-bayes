# Multinomial

### Introdução:

Esse algoritmo usa os dados em uma distribuição multinomial, que é uma generalização da distribuição binomial. Essa distribuição é parametrizada por vetores θyi=(θy1,…,θyn), θyi é a probabilidade do evento i ocorrer, dado que a classe é y. Podemos dizer que cada vetor θy representa uma observação e n é o número de features. Assim, θyi = P(xi | y) e é estimado pela seguinte fórmula:
 
![](https://lh4.googleusercontent.com/veMQe60tWgFBHXgWwTMefyLiY-xGLqxiIUiR33-ShLmIFWwbM9uwl6pmEgOgdIGiDwf0NlGwsHSIM8FL_HPDcLeYeWae7HnYjzTTAoi3FTtYbVdX3tmFZ_40RPDu_vmJhFtoH0A)

Onde Nyi é o número de vezes que a feature xi aparece no conjunto de treinamento, Ny é o número de observações com classe y, n é o número de features e alfa é uma constante que contabiliza os recursos que não estão presentes nas amostras de aprendizado e impede que haja probabilidade igual a 0. Se alfa = 1, ele é chamado de Laplace smoothing e, se alfa > 1, é chamado Lidstone smoothing. Se alfa = 0, não há correção.

### Como executar o projeto:

- Python versão 3.7 ou maior:
	- [link do site oficial](https://www.python.org/)	
- pip versão 21.1.1 ou maior:
	- [link do site oficial](https://pypi.org/)
- Pandas:
	- [link do site oficial](https://pandas.pydata.org/)
	- Comando de instalação usando pip: ```pip install pandas```
- Scikit Learn:
	- [link do site oficial](https://scikit-learn.org/);
	- Comando de instalação usando pip: ```pip install scikit-learn```

Após a instalação de todos os pacotes é só executar o comando:
```python main.py```
