import sys
import os

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import math
import numpy as np
import pandas as pd
import tensorflow as tf

from post_purpose_gru import PostPurposeGRU
from src.data.dataset_split import DatasetSplitter
from src.config import DEFAULT_DATA_FOLDER

# Load and prepare your dataset
df = pd.read_excel(os.path.join(DEFAULT_DATA_FOLDER, 'output', 'combined_posts_results.xlsx'), engine='openpyxl')

# Ensure clean_text is string and handle missing values
df['clean_text'] = df['clean_text'].astype(str).fillna('')

# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])

# Using 'total_tokens' column for dynamic padding
max_sequence_length = df['total_tokens'].max()  # Adjust maxlen dynamically
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=max_sequence_length)

# Prepare labels
labels = df[['is_analysis', 'is_evaluation', 'is_synthesis']].values

# Create a stratification column based on the target labels
df['stratify_col'] = df[['is_analysis', 'is_evaluation', 'is_synthesis']].astype(str).agg(''.join, axis=1)

# Initialize DatasetSplitter with the new stratification column
splitter = DatasetSplitter(df, 'stratify_col')
folds = splitter.split_dataset()

# Define the get_embedding_matrix function
def get_embedding_matrix(word_index, embedding_dim=200, glove_file_path=os.path.join(DEFAULT_DATA_FOLDER, 'glove.6B', 'glove.6B.200d.txt')):
    # Initialize the embedding matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    embedding_index = {}

    # Load the GloVe embeddings
    if not os.path.exists(glove_file_path):
        raise FileNotFoundError(f"{glove_file_path} not found. Please download it from https://nlp.stanford.edu/data/glove.6B.zip and extract it.")

    with open(glove_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embedding_index[word] = coefs

    # Map the GloVe embeddings to the word index
    for word, i in word_index.items():
        embedding_vector = embedding_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return embedding_matrix

# Create embedding matrix
embedding_matrix = get_embedding_matrix(tokenizer.word_index)

# Define hyperparameters
batch_size = 32
learning_rate = 0.0001
epochs = 40
gru_size = 256
hidden_size = 128
num_hidden_layers = 0

# Compile and train the model
model = PostPurposeGRU(len(tokenizer.word_index) + 1, embedding_matrix, gru_size, hidden_size, num_hidden_layers).get_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Cross-Validation Training
for train_indices, val_indices, test_indices in folds:
    X_train_fold = padded_sequences[train_indices]
    X_val_fold = padded_sequences[val_indices]
    y_train_fold = labels[train_indices]
    y_val_fold = labels[val_indices]

    model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold), batch_size=batch_size, epochs=epochs, callbacks=[early_stopping])

# Final Evaluation on Test Set
test_loss, test_accuracy = model.evaluate(padded_sequences[test_indices], labels[test_indices])
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save the model
model.save(os.path.join(DEFAULT_DATA_FOLDER, 'output', 'post_purpose_gru.h5'))
