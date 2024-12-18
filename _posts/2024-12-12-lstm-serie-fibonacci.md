--- 
layout: post
title: "LSTM Neural Networks explained for beginners with Fibonacci serie and Tensorflow"
date: 2024-12-12 00:00:00 -0300
categories: Neural-Networks AI Recurrency-Neural-Networks Deep-Learning LSTM
tag: [Artificial-Neural-Networks, Machine-Learning, Deep-Learning, AI]
---

<!--
Este sript html é necessário para a página estática do jekyll conseguir renderizar o código LaTex
-->
<script type="text/javascript" id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js">
</script>

<script>
    MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\(', '\\)']],
            displayMath: [['$$', '$$'], ['\\[', '\\]']]
        }
    };
</script>
<!--
Este sript html é necessário para a página estática do jekyll conseguir renderizar o código LaTex
-->

![Imagem de capa](../assets/Imagem_capa.png)


As **redes LSTM** (_Long Short-Term Memory_) são um tipo especial de rede neural recorrente (_RNN_) projetado para lidar com problemas relacionados a séries temporais e dados sequenciais. Elas foram introduzidas em 1997 por Sepp Hochreiter e Jürgen Schmidhuber para resolver a limitação principal das RNNs tradicionais: a incapacidade de lembrar informações de longo prazo devido ao **problema do desvanecimento** ou **explosão de gradientes** durante o treinamento.

## Estrutura e funcionamento

![Imagem da arquitetura da célula de memória](../assets/LSTM3-chain.png)

A principal inovação do LSTM é a introdução de **células de memória** e **portas** que controlam o fluxo de informações dentro da rede. Essas portas ajudam o modelo a decidir quais informações devem ser mantidas, atualizadas ou esquecidas ao longo do tempo.

- **Célula de memória:**   
    A célula de memória é responsável por armazenar informações ao longo do tempo. Ela pode reter valores por longos períodos, o que é útil para capturar dependências de longo prazo.

- **Três portas principais:**  
  - **Porta de esquecimento ($f_t$):** Decide quais informações antigas na célula de memória devem ser descartadas. É calculada por:  

$$
f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)
$$

  - **Porta de entrada ($i_t$):** Determina quais novas informações serão armazenadas na célula de memória:  

$$
i_t = \sigma(W_i \cdot [h_{t−1}, x_t] + b_i)
$$

Uma função de ativação (geralmente tangente hiperbólica) gera os valores candidatos para serem adicionados:  

$$
\tilde{C}_t = \tanh(W_C \cdot [h_{t−1} , x_t ]+ b_C )
$$

  - **Porta de saída ($o_t$):** Decide quais informações da célula de memória serão usadas para calcular a saída:  

$$
o_t​ =σ(W_o​ \cdot [h_{t−1} ,x_t ] + b_o)
$$

A saída final é modulada por uma tangente hiperbólica:  

$$
h_t = o_t \cdot \tanh(C_t)
$$

  - **Atualização da Célula de Memória ($C_t$):**
A célula de memória é atualizada com base nas portas de entrada e esquecimento:  

$$
C_t =f_t \cdot C_{t−1} + i_t \cdot \tilde C_t
​$$

