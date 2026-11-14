Especificação Projeto

1. Contextualização

Um modelo QSAR (Quantitative Structure–Activity Relationship) estabelece relações quantitativas entre a estrutura química de moléculas e uma propriedade de interesse (por exemplo, atividade biológica, permeabilidade ou toxicidade). Em termos práticos, busca-se aprender uma função que mapeia descritores da molécula para um rótulo ou valor numérico associado a ensaios laboratoriais ou curados de literatura.

Neste projeto, a molécula é representada por imagens 3D geradas a partir da estrutura (por exemplo, desenhos tipo esqueleto ou representações rasterizadas). Redes neurais convolucionais (CNNs) podem aprender padrões visuais recorrentes (subestruturas, arranjos espaciais, densidade de átomos e ligações na imagem) que correlacionam com a propriedade alvo - de forma análoga ao reconhecimento de padrões em visão computacional, porém com imagens de origem química.

Relevância: em Drug Discovery e triagem virtual (virtual screening), modelos predizem rapidamente candidatos promissores ou compostos potencialmente problemáticos, reduzindo custo experimental quando bem calibrados. O desafio é garantir generalização para moléculas fora da distribuição de treino e interpretar erros (falsos positivos/negativos) no contexto de risco e custo de ensaios.

2. Objetivo

Implementar, treinar e comparar, para cada um dos três conjuntos de dados fornecidos, duas abordagens de classificação binária com Keras (TensorFlow):

2.1 Arquitetura simples (baseline)

CNN construída do zero com no mínimo três blocos convolucionais (cada bloco pode incluir convolução, 	normalização opcional, ativação, pooling). O aluno deve justificar escolhas de número de filtros, kernel, funções de ativação, uso de Dropout/BatchNormalization (se houver), pooling e regularização.

2.2 Rede pré-treinada com fine-tuning

utilizar um backbone pré-treinado no ImageNet (exemplos aceitáveis: MobileNetV2, EfficientNetB0, 	ResNet50V2) como extrator de features, com descongelamento parcial das últimas camadas para adaptação ao domínio das imagens 	moleculares. Justificar a escolha do modelo base e a estratégia de 	descongelamento (quantas camadas, em qual ordem, se há treino em estágios).

2.3 Análise comparativa

Ao final, produzir análise comparativa entre as duas abordagens em cada dataset, discutindo trade-offs de desempenho (métricas), tempo de treinamento e complexidade de implementação/manutenção.

3. Conjuntos de dados

Os três conjuntos são tarefas de classificação binária sobre imagens já organizadas em pastas.

3.1 BACE_C

Contexto: a enzima BACE-1 (β-secretase) está ligada a processos associados à Doença de Alzheimer em linhas de pesquisa tradicionais de alvos terapêuticos. Neste dataset, a tarefa é prever se um composto se comporta como inibidor da BACE-1 ou não, a partir da imagem molecular.

Rótulos: classe binária - inibidor vs. não inibidor.

3.2 BBBP (Blood–Brain Barrier Penetration)

Contexto: a barreira hematoencefálica (BHE) controla o que entra no cérebro a partir do sangue. Saber se um composto tende a penetrar ou não a BHE é relevante para medicamentos que precisam atuar no sistema nervoso central (e também para evitar exposição cerebral indesejada).

Rótulos: classe binária - penetra a BHE vs. não penetra (conforme curadoria do conjunto).

3.3 Toxicidade

Contexto: em triagem de segurança, interessa antecipar se um composto apresenta sinalização de toxicidade (segundo o critério binário adotado na base), auxiliando priorização e desenho de ensaios.

Rótulos: classe binária - tóxico vs. não tóxico.

4. Organização dos arquivos

Os dados estão pré-processados no formato de diretórios compatível com tensorflow.keras.preprocessing.image.ImageDataGenerator ou tf.keras.utils.image_dataset_from_directory (ajuste de caminhos conforme seu ambiente):

datasets/
 ├── bace_c/
 │ ├── train/
 │ │ ├── class_0/
 │ │ └── class_1/
 │ ├── val/
 │ │ ├── class_0/
 │ │ └── class_1/
 │ └── test/
 │   ├── class_0/
 │   └── class_1/
 ├── bbbp/
 │ ├── train/
 │ │ ├── class_0/
 │ │ └── class_1/
 │ ├── val/
 │ │ ├── class_0/
 │ │ └── class_1/
 │ └── test/
 │   ├── class_0/
 │   └── class_1/
 └── toxicidade/
   ├── train/
   │ ├── class_0/
   │ └── class_1/
   ├── val/
   │ ├── class_0/
   │ └── class_1/
   └── test/
     ├── class_0/
     └── class_1/

Convenção de rótulos (class_0 e class_1) por dataset:

BACE_C - class_0: não inibidor; class_1: inibidor da BACE-1.
BBBP - class_0: não penetra a barreira hematoencefálica; class_1: penetra a BHE.
Toxicidade - class_0: não tóxico; class_1: tóxico.

5. Requisitos técnicos

Framework preferencial: Keras com backend TensorFlow (tf.keras).

Implementar as duas abordagens (baseline CNN do zero e fine-tuning com backbone ImageNet) para cada um dos três datasets - total de 6 modelos treinados e avaliados.

Fixar a mesma semente aleatória (random seed) para NumPy, TensorFlow e 	Python random em todos os experimentos, documentando o valor no notebook.

Garantir comparação justa entre os dois modelos de um mesmo dataset: idênticos tamanho de entrada (altura × largura × canais), mesma política de normalização (ex.: divisão por 255.0 e/ou 	normalização compatível com o pré-treinado, quando exigido pelo modelo base), mesmos hiperparâmetros de batch quando possível, e 	mesmo protocolo de augmentação (se usada: deve ser igual para 	ambos ou justificar por que o pré-treinado não usa augmentação 	incompatível com preprocess_input).

Obrigatório uso de callbacks EarlyStopping e ModelCheckpoint para salvar os melhores pesos com base em métrica monitorada no conjunto de validação (ex.: val_loss ou val_auc, explicitar escolha e motivo).

Salvar e versionar no relatório o melhor checkpoint carregado antes da 	avaliação no conjunto de teste.

6. Resultados esperados

Para cada um dos 6 modelos (2 abordagens × 3 datasets), reportar obrigatoriamente os itens abaixo.

6.1 Métricas de desempenho

6.1.1 Acurácias
Acurácia no conjunto de treino, validação e no conjunto de teste (separadas não basta apenas teste).

6.1.2 AUC-ROC
Valores AUC-ROC no conjunto de teste (probabilidades preditas vs. rótulo verdadeiro).

6.2.3 Matriz de confusão
Matriz de confusão no teste, com discussão de falsos positivos e falsos negativos;

6.2 Curvas de aprendizado

6.2.1 Gráficos de acurácia e loss versus época, com curvas de treino e validação, para identificar overfitting, underfitting ou instabilidade. Gráfico da evolução no treinamento e validação, com visualização da melhor época.

6.1.2 AUC-ROC
Grafico AUC-ROC no conjunto de teste (probabilidades preditas vs. rótulo verdadeiro).

6.3 Tempo de treinamento

6.3.1 Tempo época

Tempo médio por época (segundos), medido com callback customizado que registra duração por época ou com time.time() em volta de model.fit() decompondo por época (desde que o método seja explícito e reprodutível).

6.3.2 Tempo total
Tempo total até convergência considerando early stopping.

Apresentar valores em tabela comparativa baseline vs. fine-tuning para cada dataset.

6.4 Tabela comparativa final
Consolidar todos os resultados em uma única tabela com colunas sugeridas: Dataset; Modelo (Baseline / Fine-tuning); Acurácia Treino; Acurácia Teste; AUC-ROC (teste); Tempo 
médio por época (s); Tempo total (s). Seguir da tabela com análise escrita: qual abordagem foi mais vantajosa e em quais condições (dados, custo computacional, risco de overfitting, necessidade de regularização).

7. Entregáveis
Código reprodutível, contendo treino e avaliação das abordagens.

