[Versão em português](https://github.com/diegomiranda02/BERT-for-classification/blob/main/README_portugues.md)

# CREDITS
The code for this tutorial is based on a Kaggle Notebook titled "Fine-Tune BERT for Text Classification". Link to the Kaggle Notebook https://www.kaggle.com/code/pritishmishra/fine-tune-bert-for-text-classification?scriptVersionId=116951029

# A Step-by-Step Guide to Text Classification with BERT in TensorFlow
This article demonstrates a step-by-step guide on how to perform text classification using BERT, a powerful pre-trained language model, and TensorFlow. By the end, you'll be able to classify text into different categories based on your specific task. 

## Step 1: Install Dependencies
Before we dive into the code, we need to make sure we have the necessary libraries installed. You can do this using pip:

```bash
pip install transformers datasets pandas tensorflow scikit-learn
```
These libraries will help us work with BERT, manage datasets, and perform text classification efficiently.

## Step 2: Imports
We begin by importing the required libraries and modules:

```python
import pandas as pd
from datasets import DatasetDict, Dataset
from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
```

These libraries include tools for data handling, BERT model loading, tokenization, and more.

## Step 3: Read the Dataset
Assuming you have a dataset in a CSV file, we read it into a Pandas DataFrame and perform some data preprocessing:

```python
# Read the dataset from a CSV file
df = pd.read_csv("dataset.csv", sep='#', header=None)
df = df.rename(columns={0: 'descrição', 1: "label_text", 2: "text", 3: "Tipo"})
```
Here, we assume that your dataset has columns for descriptions, labels in text form, and the actual text data.

## Step 4: Mapping Labels to Integers
We need to convert the text labels into numerical values. We define a mapping function and apply it to create a new 'label' column:

```python
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


# Apply the mapping function and create a new 'label' column
df['label'] = df["label_text"].apply(map_tipo_to_int)
```

This ensures our labels are in a format suitable for training a machine learning model.

## Step 5: Data Preprocessing
We perform some preprocessing on the text data, such as converting it to lowercase and truncating it to a maximum length:

```python
# Convert text to lowercase
df['label_text'] = df['label_text'].str.lower()
df['text'] = df['text'].str.lower()

# Truncate text to a maximum of 128 characters
df['text'] = df['text'].apply(lambda x: truncate_text(x, max_length=128))
```
These steps help standardize the text data and manage its length.

## Step 6: Creating Dataset Objects
We convert our Pandas DataFrame into Dataset objects:

```python
# Convert DataFrames to Dataset objects
df_dataset = Dataset.from_pandas(df)

# Remove unnecessary columns
df_dataset = df_dataset.remove_columns('__index_level_0__')
```
This allows us to work efficiently with the data using the Hugging Face datasets library.

## Step 7: Splitting the Dataset
We split the dataset into training and testing sets:

```python
# Split the DataFrame into training and test sets (e.g., 80% train, 20% test)
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Create Dataset objects for training and testing
train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns('__index_level_0__')
test_dataset = test_dataset.remove_columns('__index_level_0__')

# Combine datasets into a DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'test': test_dataset,
})
```
This separation ensures we have a set of data for training the model and another for evaluating its performance.

## Step 8: Loading BERT Model and Tokenizer
We load the pre-trained BERT model and its corresponding tokenizer:

```python
model = TFAutoModel.from_pretrained("neuralmind/bert-base-portuguese-cased")
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
```
We'll use this BERT model for text classification pre-trained in Brazilian Portuguese language.

## Step 9: Tokenization and Encoding
We tokenize and encode our text data for use with the BERT model:

```python
def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

# Tokenize and encode the dataset
dataset_encoded = dataset_dict.map(tokenize, batched=True, batch_size=None)

# Set the format to TensorFlow
dataset_encoded.set_format('tf', columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])
```
This prepares the data in a format compatible with TensorFlow.

## Step 10: Mixed Precision Training (Optional)
We can enable mixed-precision training for faster model training on compatible hardware:

```python
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```
This is an optional step but can significantly speed up training if you have the necessary hardware.

## Step 11: Building the TensorFlow Datasets
We create TensorFlow datasets for training and testing:

```python
BATCH_SIZE = 64

def order(inp):
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

# Create the training dataset
train_dataset = tf.data.Dataset.from_tensor_slices(dataset_encoded['train'][:])
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(1000)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

# Create the testing dataset
test_dataset = tf.data.Dataset.from_tensor_slices(dataset_encoded['test'][:])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)
```
These TensorFlow datasets are what we'll use for training and evaluation.

## Step 12: Building the Classification Model
We define our classification model, which consists of the BERT model and a dense layer for class prediction:

```python
class BERTForClassification(tf.keras.Model):

    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        return self.fc(x)

classifier = BERTForClassification(model, num_classes=6)
```
This model will be trained to classify text into one of six classes.

## Step 13: Compiling and Training the Model
We compile and train the classification model:

```python
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
```
This step involves specifying the optimizer, loss function, and metrics for training.

## Step 14: Evaluating the Model
We evaluate the model's performance on the test dataset:

```python
classifier.evaluate(test_dataset)
```
This provides us with metrics like accuracy to assess how well the model generalizes to new data.

## Step 15: Making Predictions
Finally, we use our trained model to make predictions on new text data:

```python
input_text = "Processo nº 48500.003134/2012-10. Interessado: Migratio Gestão e Comercialização de Energia Elétrica Ltda. Decisão: Autorizar a empresa Migratio Gestão e Comercialização de Energia Elétrica Ltda., inscrita no CNPJ/MF sob nº 15.458.171/0001-36, a atuar como Agente Comercializador de Energia Elétrica no âmbito da CCEE; informar que a atividade poderá ser exercida por meio de sua filial, CNPJ/MF sob nº 15.458.171/0002-17."

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
predicted_class = np.argmax(predictions, axis=-1)[0]

# Print the predicted class
print("Predicted Class:", predicted_class)
```
This demonstrates how to use your trained model for real-world text classification tasks.

## Dataset
The dataset was downloaded from https://www.in.gov.br/acesso-a-informacao/dados-abertos/base-de-dados and processed using the script ---.py and ----.py.

## References
This tutorial draws from various online resources and the official documentation of the libraries used:

Hugging Face Transformers Library
Hugging Face Datasets Library
TensorFlow Official Documentation
Scikit-Learn Documentation


