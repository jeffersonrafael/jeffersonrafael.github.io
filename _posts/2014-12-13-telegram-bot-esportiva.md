---
layout: post
title: "Diário dev #1 | Monitore jogos de futebol e receba sinais de arbitragem esportiva em tempo real no Telegram com Python"
date: 2024-12-14 00:00:00 -0300
categories: Telegram-Bot Arbitragem-Esportiva Python API
tag: [Telegram-Bot, Python]
---


## Apresentando o projeto

<!-- Eu estou construindo um robô de arbitragem esportiva no Telegram para monitorar resultados e estatísticas de jogos de futebol em tempo real e enviar sinais de oportunidade de arbitragem esportiva a partir de previsões feitas por modelos de deep learning e inteligência artificial. -->

Estou desenvolvendo um bot de arbitragem esportiva no Telegram, projetado para o monitoramento de resultados e estatísticas de jogos de futebol em tempo real. Este robô combina a precisão de modelos avançados de deep learning e inteligência artificial para identificar oportunidades de arbitragem esportiva, enviando sinais estratégicos que transformam dados em vantagem competitiva. __Prepare-se para elevar o jogo e explorar o futuro da análise esportiva como nunca antes!__

O Bot Telegram está integrado numa API para extrair dados de partidas de futebol de campeonatos do Brasil e do Mundo em tempo real. Eu decidi começar o projeto construindo um [PoC](https://pt.wikipedia.org/wiki/Prova_de_conceito) funcional do meu Bot Telegram, mostrando como seria o seu funcionamento. 

A ideia inicial é criar um sistema que colete informações de partidas, como placar, estatísticas de desempenho e odds oferecidas por casas de apostas, processando esses dados em um modelo treinado para identificar oportunidades que possam ser exploradas para arbitragem. Essa funcionalidade permitirá que o bot envie alertas diretamente para os usuários interessados, possibilitando decisões rápidas e estratégicas.

O objetivo principal do PoC é validar a viabilidade técnica do projeto, demonstrar a integração eficaz com APIs esportivas e comprovar o potencial dos algoritmos de inteligência artificial aplicados. Uma vez testada e refinada essa versão inicial, pretendo expandir o sistema com mais funcionalidades, como personalização de alertas, análises preditivas de resultados e suporte a outros esportes, tornando-o uma ferramenta robusta e eficiente para entusiastas de arbitragem esportiva.

<!-- ![Demonstração do PoC](../assets/video_2024-12-14_13-45-32.gif) -->

## Desafios técnicos

Era dia 10 de dezembro, e eu se preparava para mais um dia de trabalho. Mas, ao se deparar com o código do Bot do Telegram, algo não estava certo. A aplicação, que deveria retornar mensagens de forma simples, se recusava a cooperar. A lógica parecia estar correta, as linhas de código estavam bem escritas, mas o Bot simplesmente não funcionava como esperado.

Eu havia usado decarators para evitar utilizar um monte de _If_ e _elif_ para construir os botões, como no código abaixo:
```python
@botao("H2H")
def handle_h2h(update, context):
    keyboard = [
        [InlineKeyboardButton("🔥Clube Black🔥", callback_data="VIP")],
        [InlineKeyboardButton(f"🔝{texto_voltar}🔝", callback_data="BACK")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    update.callback_query.edit_message_text(
        "Informações sobre os últimos confrontos das equipes", reply_markup=reply_markup
        )
```

No início, a frustração tomou conta. A cada tentativa, o mesmo erro surgia, e as soluções que pareciam promissoras acabavam em novas dúvidas. As horas passavam, e eu permanecia fixo na cadeira, encarando o monitor como um adversário. Enquanto o sol se punha, a sensação de cansaço começava a pesar, mas desistir não era uma opção.

Com determinação, eu revisei o código mais uma vez, linha por linha, examinando cada detalhe. A persistência começou a transformar o estresse em foco. Cada erro identificado, cada ajuste feito era uma pequena vitória. Eu estava aprendendo a lidar com a pressão, a transformar a frustração em método.

Finalmente, antes que o dia terminasse, a solução veio. Era simples, mas brilhante. O Bot não só funcionava, mas rodava de forma perfeita, fluída, como eu sempre imaginei. Naquele momento, percebi que o esforço valera a pena.

<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Centralizar GIF</title>
    <style>

        .gif-container {
            text-align: center;
        }

        .gif-container img {
            width: 200px; /* Defina a largura desejada */
            height: 435px; /* Defina a altura desejada */
        }
</style>
</head>
<body>
    <div class="gif-container">
        <img src="../assets/video_2024-12-14_13-45-32.gif" alt="Esse gif apresenta uma demonstração do conceito do Bot Telegram construído até agora.">
    </div>
</body>
</html>

_Esse gif apresenta uma demonstração da Prova de Conceito (PoC) do Bot Telegram construído até agora._ 

O problema que parecia insuperável pela manhã tornou-se o símbolo de sua superação à noite. Eu me levantei da cadeira não apenas com o código resolvido, mas com a certeza de que havia crescido. Aprendi que, às vezes, o maior obstáculo não é o erro em si, mas a capacidade de manter a mente firme até encontrar a resposta.

Neste projeto, eu faço a integração do Bot Telegram com três APIs externas:
- A API do site de dados esportivos. Onde eu faço a requisição para extrair os dados.
- A API do Telegram para construir o Bot personalizado para o envio de alertas e integração com o sistema de monitoramento e inteligência artificial.
- A API do Mercado Pago para construir o sistema de pagamento da aplicação.



Os próximos passos será a integração das APIs com um banco de dados. Este é o ponto de partida do projeto, e sua primeira versão já está disponível no meu [github](https://github.com/jeffersonrafael/FutebolAPI-Bot). Se você tem interesse no projeto e gostaria de ser atualizado sobre os avanços nele, te convido a clicar em __watch__ ou em __star__ no repositório.
