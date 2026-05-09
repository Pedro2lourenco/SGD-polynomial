# Regressão Polinomial com SGD e Gerador Pseudoaleatório XORSHIRO

Este projeto implementa uma regressão polinomial cúbica utilizando o algoritmo de Gradiente Descendente Estocástico (SGD) em Python.
Os dados são gerados artificialmente com ruído aleatório produzido por um gerador pseudoaleatório baseado no algoritmo XORSHIRO.

O código também exibe uma animação em tempo real do processo de treinamento do modelo.

---

# Objetivo

Ajustar um polinômio cúbico aos dados gerados pela função:

genui{"math_block_widget_always_prefetch_v2":{"content":"y = 3x^3 - 2x^2 + x + 3"}}

utilizando SGD para aprender os coeficientes:

\hat{y} = ax^3 + bx^2 + cx + d

---

# Bibliotecas Utilizadas

```python
numpy
matplotlib
```

Instalação:

```bash
pip install numpy matplotlib
```

---

# Estrutura do Código

## 1. Configuração do Matplotlib

Define o estilo visual dos gráficos:

* Fonte serifada
* Eixos mais espessos
* Ticks internos
* Legenda ativada

---

## 2. Gerador Pseudoaleatório XORSHIRO

O código implementa um gerador pseudoaleatório baseado no algoritmo XORSHIRO.

Funções principais:

```python
rotl(x, k)
```

Realiza rotação de bits.

```python
next_xorshiro()
```

Gera o próximo número pseudoaleatório de 64 bits.

```python
rand_uniform()
```

Converte o valor gerado em um número de ponto flutuante no intervalo:

[0,1)

---

# Geração dos Dados

Os dados são gerados no intervalo:

x \in [-3,3]

com:

```python
x = np.linspace(-3,3,100)
```

A função verdadeira é:

```python
y = 3*x**3 - 2*x**2 + x + 3
```

Adiciona-se ruído aleatório:

```python
y += rand_uniform()
```

---

# Inicialização dos Parâmetros

Os coeficientes do modelo são inicializados aleatoriamente:

```python
a,b,c,d = rand_uniform(),rand_uniform(),rand_uniform(),rand_uniform()
```

---

# Treinamento com SGD

O treinamento utiliza:

* Learning rate:

\eta = 5 \times 10^{-4}

* Número de épocas:

```python
epchs = 20
```

---

# Função de Erro

O erro é calculado por:

error = y - \hat{y}

---

# Atualização dos Parâmetros

Os gradientes utilizados são:

\frac{\partial L}{\partial a} = -2x^3(error)

\frac{\partial L}{\partial b} = -2x^2(error)

\frac{\partial L}{\partial c} = -2x(error)

\frac{\partial L}{\partial d} = -2(error)

Atualização:

\theta = \theta - \eta \nabla L

---

# Visualização

Aqui está a versão ajustada, mantendo o link para o seu GIF de visualização e aplicando as correções matemáticas com LaTeX:
# Regressão Polinomial via SGD
Este projeto implementa o ajuste de uma curva polinomial de terceiro grau utilizando o algoritmo de Gradiente Descendente Estocástico (SGD). O objetivo é aproximar os parâmetros de uma função teórica a partir de dados observados.
## Configurações Experimentais
 * **Taxa de Aprendizado (\eta):** 5 \times 10^{-4}
 * **Épocas:** 20
 * **Modelo:** \hat{y} = ax^3 + bx^2 + cx + d
## Formulação Matemática
O erro instantâneo é definido pela diferença entre o valor real e a predição:

### Gradientes da Função de Custo
Os gradientes em relação aos coeficientes (derivados da função de perda quadrática) são:
 *  *  *  * ### Regra de Atualização
Os parâmetros \theta = \{a, b, c, d\} são atualizados iterativamente:

## Visualização do Treinamento
Abaixo, a animação mostra a curva prevista convergindo para os dados originais em tempo real:
<p align="center">
<img src="sgd_training.gif" width="500">
</p>
## Objetivo de Convergência
O algoritmo busca reduzir o erro para que o modelo aproxime-se da função alvo:

## Tecnologias e Conceitos
 * **Otimização:** Gradiente Descendente Estocástico (SGD).
 * **Numérico:** Ajuste de curvas e geradores pseudoaleatórios (XORSHIRO).
 * **Visualização:** Matplotlib para renderização dinâmica.
## Próximos Passos
 * Implementação de Minibatch Gradient Descent.
 * Uso de otimizadores avançados (Adam ou Momentum).
 * Inclusão de métricas como Erro Quadrático Médio (MSE).
