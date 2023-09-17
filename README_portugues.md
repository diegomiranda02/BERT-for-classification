[English Version](https://github.com/diegomiranda02/BERT-for-classification/blob/main/README.md)

## Créditos
O código deste tutorial é baseado em um Notebook do Kaggle intitulado "Fine-Tune BERT for Text Classification". Link para o Notebook do Kaggle [aqui](https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029)

## Um Guia Passo a Passo para Classificação de Texto com BERT no TensorFlow
Este artigo demonstra um guia passo a passo sobre como realizar a classificação de texto usando BERT, um poderoso modelo de linguagem pré-treinado, e o TensorFlow. Ao final deste guia, você será capaz de classificar texto em diferentes categorias com base em sua tarefa específica.

## Passo 1: Instalar Dependências
Antes de iniciar no código, é importante garantir que as bibliotecas necessárias estejam instaladas. Você pode fazer isso usando o pip:

```bash
pip install transformers datasets pandas tensorflow scikit-learn
```
Essas bibliotecas nos ajudarão a trabalhar com o BERT, gerenciar conjuntos de dados e realizar classificação de texto de forma eficiente.

## Passo 2: Importações
Começamos importando as bibliotecas e módulos necessários:

```python
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
```
Essas bibliotecas incluem ferramentas para manipulação de dados, carregamento de modelos BERT, tokenização e muito mais.

## Passo 3: Ler o Conjunto de Dados
Supondo que você tenha um conjunto de dados em um arquivo CSV, lemos esse conjunto de dados em um DataFrame do Pandas e realizamos algum pré-processamento de dados:

```python
# Ler o conjunto de dados de um arquivo CSV
df = pd.read_csv("dataset.csv", sep='#', header=None)
df = df.rename(columns={0: 'descrição', 1: "label_text", 2: "text", 3: "Tipo"})
```
Aqui, supomos que seu conjunto de dados tenha colunas para descrições, rótulos em formato de texto e os próprios dados de texto.

## Passo 4: Mapear Rótulos para Inteiros
Precisamos converter os rótulos de texto em valores numéricos. Definimos uma função de mapeamento e a aplicamos para criar uma nova coluna 'label':

```python
# Função para mapear valores do Tipo para inteiros (sem diferenciação de maiúsculas e minúsculas)
def map_tipo_to_int(tipo):
    tipo_lower = tipo.lower()
    if 'despacho' in tipo_lower:
        return 1
    elif 'portaria' in tipo_lower:
        return 2
    elif 'resolução-re' in tipo_lower:
        return 3
    else:
        return tipo

# Aplicar a função de mapeamento e criar uma nova coluna 'label'
df['label'] = df["label_text"].apply(map_tipo_to_int)
```
Isso garante que nossos rótulos estejam em um formato adequado para treinar um modelo de aprendizado de máquina.

## Passo 5: Pré-processamento de Dados
Realizamos algum pré-processamento nos dados de texto, como converter tudo para letras minúsculas e truncar o texto com um comprimento máximo:

```python
# Converter texto para minúsculas
df['label_text'] = df['label_text'].str.lower()
df['text'] = df['text'].str.lower()

# Truncar o texto com um máximo de 128 caracteres
df['text'] = df['text'].apply(lambda x: truncate_text(x, max_length=128))
```
Essas etapas ajudam a padronizar os dados de texto e gerenciar seu comprimento.

## Passo 6: Criar Objetos de Conjunto de Dados
Convertemos nosso DataFrame do Pandas em objetos de Conjunto de Dados:

```python
# Converter DataFrames em objetos de Conjunto de Dados
df_dataset = Dataset.from_pandas(df)

# Remover colunas desnecessárias
df_dataset = df_dataset.remove_columns('__index_level_0__')
```
Isso nos permite trabalhar de forma eficiente com os dados usando a biblioteca de Conjunto de Dados da Hugging Face.

## Passo 7: Dividir o Conjunto de Dados
Dividimos o conjunto de dados em conjuntos de treinamento e teste:

```python
# Dividir o DataFrame em conjuntos de treinamento e teste (por exemplo, 80% de treinamento, 20% de teste)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Criar objetos de Conjunto de Dados para treinamento e teste
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Remover colunas desnecessárias
train_dataset = train_dataset.remove_columns('__index_level_0__')
test_dataset = test_dataset.remove_columns('__index_level_0__')

# Combinar conjuntos de dados em um DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
})
```
Essa separação garante que tenhamos um conjunto de dados para treinar o modelo e outro para avaliar seu desempenho.

## Passo 8: Carregar o Modelo BERT e o Tokenizador
Carregamos o modelo BERT pré-treinado e seu tokenizador correspondente:

```python
model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
```
Vamos usar este modelo BERT para classificação de texto pré-treinado na língua portuguesa do Brasil.

## Passo 9: Tokenização e Codificação
Tokenizamos e codificamos nossos dados de texto para uso com o modelo BERT:

```python
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Tokenizar e codificar o conjunto de dados
dataset_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)

# Definir o formato para TensorFlow
dataset_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
```
Isso prepara os dados em um formato compatível com o TensorFlow.

##Passo 10: Treinamento com Precisão Mista (Opcional)
Podemos habilitar o treinamento com precisão mista para treinar o modelo mais rapidamente em hardware compatível:

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```
Este é um passo opcional, mas pode acelerar significativamente o treinamento se você tiver o hardware necessário.

## Passo 11: Construir os Conjuntos de Dados do TensorFlow
Criamos conjuntos de dados do TensorFlow para treinamento e teste:

```python
BATCH_SIZE = 64

def order(inp):
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

# Criar o conjunto de dados de treinamento
train_dataset = tf.data.Dataset.from_tensor_slices(dataset_encoded['train'][:])
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

# Criar o conjunto de dados de teste
test_dataset = tf.data.Dataset.from_tensor_slices(dataset_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)
```
Esses conjuntos de dados TensorFlow são o que usaremos para treinamento e avaliação.

## Passo 12: Construir o Modelo de Classificação
Definimos nosso modelo de classificação, que consiste no modelo BERT e em uma camada densa para previsão de classe:

```python
class BERTParaClassificacao(tf.keras.Model):

    def __init__(self, modelo_bert, num_classes):
        super().__init__()
        self.bert = modelo_bert
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, entradas):
        x = self.bert(entradas)[1]
        return self.fc(x)

classificador = BERTParaClassificacao(model, num_classes=6)
```
Este modelo será treinado para classificar texto em uma das seis classes.

## Passo 13: Compilar e Treinar o Modelo
Compilamos e treinamos o modelo de classificação:

```python
classificador.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

histórico = classificador.fit(
    train_dataset,
    epochs=3,
    batch_size=4
)
```
Este passo envolve a especificação do otimizador, função de perda e métricas para o treinamento.

## Passo 14: Avaliar o Modelo
Avaliamos o desempenho do modelo no conjunto de dados de teste:

```python
classificador.evaluate(test_dataset)
```
Isso nos fornece métricas como a precisão para avaliar o quão bem o modelo generaliza para novos dados.

## Passo 15: Fazer Previsões
Finalmente, usamos nosso modelo treinado para fazer previsões em novos dados de texto:

```python
texto_de_entrada = "Processo nº 48500.003134/2012-10. Interessado: Migratio Gestão e Comercialização de Energia Elétrica Ltda. Decisão: Autorizar a empresa Migratio Gestão e Comercialização de Energia Elétrica Ltda., inscrita no CNPJ/MF sob nº 15.458.171/0001-36, a atuar como Agente Comercializador de Energia Elétrica no âmbito da CCEE; informar que a atividade poderá ser exercida por meio de sua filial, CNPJ/MF sob nº 15.458.171/0002-17."

# Tokenizar e pré-processar a instância de entrada
tokens_de_entrada = tokenizer(texto_de_entrada, padding=True, truncation=True)

# Tokenizar e pré-processar o texto de entrada
input_ids = tokens_de_entrada['input_ids']
attention_mask = tokens_de_entrada['attention_mask']
token_type_ids = tokens_de_entrada['token_type_ids']

# Converter a entrada pré-processada em um tensor TensorFlow
input_ids = tf.convert_to_tensor([input_ids])
attention_mask = tf.convert_to_tensor([attention_mask])
token_type_ids = tf.convert_to_tensor([token_type_ids])

# Fazer previsões usando o classificador
previsões = classificador.predict([input_ids, attention_mask, token_type_ids])

# Obter a classe prevista
classe_prevista = np.argmax(previsões, axis=-1)[0]

# Imprimir a classe prevista
print("Classe Prevista:", classe_prevista)
```
Isso demonstra como usar seu modelo treinado para tarefas reais de classificação de texto.

Conjunto de Dados
O conjunto de dados foi baixado em https://www.in.gov.br/acesso-a-informacao/dados-abertos/base-de-dados e processado usando os scripts ---.py e ----.py.
