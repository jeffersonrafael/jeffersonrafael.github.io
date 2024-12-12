--- 
title: "LSTM Neural Networks explained for beginners with Fibonacci serie and Tensorflow"
date: 2024-12-12 00:00:00 -0300
categories: Neural-Networks AI Recurrency-Neural-Networks Deep-Learning LSTM
tag: [Artificial-Neural-Networks, Machine-Learning, Deep-Learning, AI]
---



![Imagem de capa](../assets/Imagem_capa.png)


As **redes LSTM** (__Long Short-Term Memory__) são um tipo especial de rede neural recorrente (RNN) projetado para lidar com problemas relacionados a séries temporais e dados sequenciais. Elas foram introduzidas em 1997 por Sepp Hochreiter e Jürgen Schmidhuber para resolver a limitação principal das RNNs tradicionais: a incapacidade de lembrar informações de longo prazo devido ao **problema do desvanecimento** ou **explosão de gradientes** durante o treinamento.

## Estrutura e funcionamento

![Imagem de capa](../assets/LSTM3-chain.png)

A principal inovação do LSTM é a introdução de **células de memória** e **portas** que controlam o fluxo de informações dentro da rede. Essas portas ajudam o modelo a decidir quais informações devem ser mantidas, atualizadas ou esquecidas ao longo do tempo.

- **Célula de memória:**   
    A célula de memória é responsável por armazenar informações ao longo do tempo. Ela pode reter valores por longos períodos, o que é útil para capturar dependências de longo prazo.

- **Três portas principais:**  
  1. **Porta de esquecimento (ft):** Decide quais informações antigas na célula de memória devem ser descartadas. É calculada por:
$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$