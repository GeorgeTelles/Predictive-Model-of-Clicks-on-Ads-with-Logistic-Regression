<div>
  <img src="https://raw.githubusercontent.com/GeorgeTelles/georgetelles/f69531ec6b293b5148563588a764c010015d315e/logo_clara.png" alt="logo clara" width="300" style="display: inline-block; vertical-align: top; margin-right: 10px;">
  <img src="https://raw.githubusercontent.com/GeorgeTelles/georgetelles/f69531ec6b293b5148563588a764c010015d315e/logo_dark.png" alt="logo dark" width="300" style="display: inline-block; vertical-align: top;">
</div>

# Predictive Model of Clicks on Ads with Logistic Regression




Neste projeto será analisado um conjunto de dados de publicidade, indicando se um usuário de internet específico clicou ou não em um anuncio. O objetivo é criar um modelo que preveja se clicará ou não em um anúncio baseado nos dados desse usuário.

Este conjunto de dados contém os seguintes recursos:

* 'Daily Time Spent on Site': tempo no site em minutos.
* 'Age': idade do consumidor.
* 'Area Income': Média da renda do consumidor na região.
* 'Daily Internet Usage': Média em minutos por di que o consumidor está na internet.
* 'Linha do tópico do anúncio': Título do anúncio.
* 'City': Cidade do consumidor.
* 'Male': Se o consumidor era ou não masculino.
* 'Country': País do consumidor.
* 'Timestamp': hora em que o consumidor clicou no anúncio ou janela fechada.
* 'Clicked on Ad'': 0 ou 1 indicam se clicou ou não no anúncio.

# Resumo dos Insights e Métricas

## 1. Relatório de Classificação

O **relatório de classificação** oferece uma visão detalhada da performance do modelo com base em várias métricas importantes:

### **Precisão (Precision)**

- **Definição**: A precisão mede a proporção de verdadeiros positivos em relação ao total de previsões positivas feitas pelo modelo. Em outras palavras, é a taxa de acerto entre todas as previsões de uma determinada classe.
  
  \[
  \text{Precisão} = \frac{\text{Verdadeiros Positivos (TP)}}{\text{Verdadeiros Positivos (TP)} + \text{Falsos Positivos (FP)}}
  \]

- **Resultado**:
  - **Classe 0 (Não Clicou)**: 0.86
    - Significa que, entre todas as previsões de "Não Clicou", 86% estavam corretas. O modelo teve 14% de previsões falsas para essa classe.
  - **Classe 1 (Clicou)**: 0.96
    - Significa que, entre todas as previsões de "Clicou", 96% estavam corretas. O modelo teve apenas 4% de previsões falsas para essa classe.

  A alta precisão para a classe "Clicou" sugere que o modelo é muito eficaz em identificar corretamente quando um usuário clicou no anúncio, enquanto a precisão mais baixa para a classe "Não Clicou" indica que há alguns erros na identificação de não cliques.

### **Revocação (Recall)**

- **Definição**: A revocação mede a proporção de verdadeiros positivos identificados corretamente em relação ao total de casos reais positivos. Em outras palavras, é a taxa de detecção dos casos positivos verdadeiros.

  \[
  \text{Revocação} = \frac{\text{Verdadeiros Positivos (TP)}}{\text{Verdadeiros Positivos (TP)} + \text{Falsos Negativos (FN)}}
  \]

- **Resultado**:
  - **Classe 0 (Não Clicou)**: 0.96
    - Significa que, entre todos os reais "Não Clicou", 96% foram corretamente identificados pelo modelo. Isso indica uma excelente capacidade de identificar usuários que não clicaram no anúncio.
  - **Classe 1 (Clicou)**: 0.85
    - Significa que, entre todos os reais "Clicou", 85% foram corretamente identificados pelo modelo. Há uma pequena proporção de cliques verdadeiros que o modelo não conseguiu detectar.

  A revocação mais alta para "Não Clicou" mostra que o modelo é muito bom em identificar esses casos, enquanto a revocação para "Clicou" ainda é boa, mas não tão alta quanto para "Não Clicou".

### **Pontuação F1 (F1 Score)**

- **Definição**: A Pontuação F1 é a média harmônica entre a precisão e a revocação. Ela combina ambas as métricas em um único valor, útil quando há a necessidade de balancear precisão e revocação.

  \[
  \text{Pontuação F1} = 2 \times \frac{\text{Precisão} \times \text{Revocação}}{\text{Precisão} + \text{Revocação}}
  \]

- **Resultado**:
  - **Classe 0 (Não Clicou)**: 0.91
    - A média harmônica entre precisão (0.86) e revocação (0.96) para a classe "Não Clicou".
  - **Classe 1 (Clicou)**: 0.90
    - A média harmônica entre precisão (0.96) e revocação (0.85) para a classe "Clicou".

  A Pontuação F1 é bastante equilibrada para ambas as classes, mostrando que o modelo tem um bom equilíbrio entre identificar corretamente os cliques e não cliques.

### **Acurácia (Accuracy)**

- **Definição**: A acurácia é a proporção total de previsões corretas (tanto positivas quanto negativas) em relação ao total de previsões feitas. É uma medida geral da eficácia do modelo.

  \[
  \text{Acurácia} = \frac{\text{Verdadeiros Positivos (TP)} + \text{Verdadeiros Negativos (TN)}}{\text{Total de Amostras}}
  \]

- **Resultado**: 0.91
  - O modelo acertou 91% das previsões, indicando um desempenho geral muito bom.

### **Média Macro e Ponderada**

- **Macro Avg**:
  - **Definição**: A média não ponderada das métricas de precisão, revocação e F1 Score para todas as classes.
  - **Resultado**: 0.91
    - É a média das métricas para cada classe, tratando todas as classes igualmente.

- **Weighted Avg**:
  - **Definição**: A média ponderada das métricas de precisão, revocação e F1 Score, levando em consideração o número de amostras de cada classe.
  - **Resultado**: 0.91
    - Reflete a média das métricas ajustada pelo número de amostras em cada classe, oferecendo uma visão mais equilibrada considerando a distribuição das classes.

## 2. Matriz de Confusão

A **matriz de confusão** ajuda a visualizar a performance do modelo com relação aos verdadeiros positivos (TP), verdadeiros negativos (TN), falsos positivos (FP) e falsos negativos (FN):

- **Verdadeiros Negativos (TN)**: 156
  - O número de casos em que o modelo corretamente previu "Não Clicou".

- **Falsos Positivos (FP)**: 6
  - O número de casos em que o modelo previu "Clicou", mas o real foi "Não Clicou".

- **Falsos Negativos (FN)**: 25
  - O número de casos em que o modelo previu "Não Clicou", mas o real foi "Clicou".

- **Verdadeiros Positivos (TP)**: 143
  - O número de casos em que o modelo corretamente previu "Clicou".

### Interpretação dos Resultados

- **Precisão** é mais alta para a classe "Clicou" (0.96) do que para "Não Clicou" (0.86). Isso sugere que o modelo é muito eficaz em identificar casos onde o usuário de fato clicou no anúncio, mas é um pouco menos preciso em identificar corretamente os casos onde o usuário não clicou.

- **Revocação** é mais alta para a classe "Não Clicou" (0.96) do que para "Clicou" (0.85). Isso indica que o modelo é melhor em identificar usuários que não clicaram nos anúncios do que em identificar aqueles que realmente clicaram.

- **Pontuação F1** é similar para ambas as classes (0.91 para "Não Clicou" e 0.90 para "Clicou"), mostrando um bom equilíbrio entre precisão e revocação.

- A **matriz de confusão** confirma que o modelo tem um desempenho geral muito bom, com a maior parte das previsões corretas, mas há alguns casos de falsos positivos e negativos que ainda precisam ser ajustados.

## 3. Análise Visual

- **Pairplot**: O pairplot mostra a relação entre os diferentes recursos e como eles se distribuem entre as classes "Clicou" e "Não Clicou". Ajuda a entender como cada recurso contribui para a separação das classes e identificar padrões ou correlações.
