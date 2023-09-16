# Install the dependencies
pip install transformers datasets

# Imports
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split


# Read the dataset
df = pd.read_csv("/content/drive/MyDrive/Projetos/NLP/Sinapses_Classificador/dataset/dataset_gold.csv", sep='#', header=None)
df = df.rename(columns={0: 'descrição', 1: "label_text", 2: "text", 3: "Tipo"})

# Function to map Tipo values to integers (case-insensitive)
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

# Apply the mapping function to the 'Tipo' column and create a new 'Mapped_Tipo' column
df['label'] = df["label_text"].apply(map_tipo_to_int)

df = df[["text","label","label_text"]]
# Convert values in the 'text' and 'label_text' columns to lowercase
df['label_text'] = df['label_text'].str.lower()
df['text'] = df['text'].str.lower()

df = df.dropna(subset=['text'])

# Define a function to truncate text to a maximum of 512 characters
def truncate_text(text, max_length=512):
    return text[:max_length]

# Apply the truncate_text function to the 'text_column' and store the result in a new column
df['truncated_text'] = df['text'].apply(lambda x: truncate_text(x, max_length=128))

df = df[["label","label_text","truncated_text"]]
df = df.rename(columns={'truncated_text': 'text'})

# Convert DataFrames to Dataset objects
df_dataset = Dataset.from_pandas(df)

# Remove the '__index_level_0__' column
df_dataset = df_dataset.remove_columns('__index_level_0__')

# Split the DataFrame into training and test sets (e.g., 80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create Dataset for training data
train_dataset = Dataset.from_pandas(train_df)

# Create Dataset for testing data
test_dataset = Dataset.from_pandas(test_df)

# Remove the '__index_level_0__' column
train_dataset = train_dataset.remove_columns('__index_level_0__')
test_dataset = test_dataset.remove_columns('__index_level_0__')

dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
})

model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

def tokenize(batch):
  return tokenizer(batch["text"], padding=True, truncation=True)

dataset_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)

dataset_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

BATCH_SIZE = 64

def order(inp):
  data = list(inp.values())
  return {
      'input_ids':data[1],
      'attention_mask':data[2],
      'token_type_ids':data[3]
  }, data[0]

# converting train split of 'emotions_encoded' to tensorflow format
train_dataset = tf.data.Dataset.from_tensor_slices(dataset_encoded['train'][:])
# set batch_size and shuffle
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
# map the 'order' function
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

# .. doing the same for test set ...
test_dataset = tf.data.Dataset.from_tensor_slices(dataset_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

class BERTForClassification(tf.keras.Model):

  def __init__(self, bert_model, num_classes):
    super().__init__()
    self.bert = bert_model
    self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

  def call(self, inputs):
    x = self.bert(inputs)[1]
    return self.fc(x)

classifier = BERTForClassification(model, num_classes=6)

classifier.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

history = classifier.fit(
    train_dataset,
    epochs=3,
    batch_size=4
)

classifier.evaluate(test_dataset)

# Define a single input instance
input_text = "Processo nº 48500.003134/2012-10. Interessado: Migratio Gestão e Comercialização de Energia Elétrica Ltda. Decisão: Autorizar a empresa Migratio Gestão e Comercialização de Energia Elétrica Ltda., inscrita no CNPJ/MF sob nº 15.458.171/0001-36, a atuar como Agente Comercializador de Energia Elétrica no âmbito da CCEE; informar que a atividade poderá ser exercida por meio de sua filial, CNPJ/MF sob nº 15.458.171/0002-17. A íntegra deste Despacho consta dos autos e encontra-se disponível em www.aneel.gov.br/biblioteca."

# Tokenize and preprocess the input instance
input_tokens = tokenizer(input_text, padding=True, truncation=True)

# Tokenize and preprocess the input text
input_ids = input_tokens['input_ids']
attention_mask = input_tokens['attention_mask']
token_type_ids = input_tokens['token_type_ids']

# Convert the preprocessed input into a TensorFlow tensor
input_ids = tf.convert_to_tensor([input_ids])
attention_mask = tf.convert_to_tensor([attention_mask])
token_type_ids = tf.convert_to_tensor([token_type_ids])

# Make predictions using the classifier
predictions = classifier.predict([input_ids, attention_mask, token_type_ids])

# Get the predicted class
predicted_class = np.argmax(predictions, axis=-1)[0]  # Assuming you have a single instance

# Print the predicted class
print("Predicted Class:", predicted_class)


