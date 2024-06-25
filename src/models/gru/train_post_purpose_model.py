# pylint: disable=no-member

"""
This model includes the training of the GRU model.
"""


import os
import sys
import tensorflow as tf
import pandas as pd
import numpy as np
from src.config import DEFAULT_DATA_FOLDER
from src.data.dataset_split import DatasetSplitter
from src.models.gru.post_purpose import PostPurposeGRU

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..')))


# Load and prepare your dataset
df = pd.read_excel(os.path.join(DEFAULT_DATA_FOLDER, 'output',
                   'combined_posts_results.xlsx'), engine='openpyxl')

# Ensure clean_text is string and handle missing values
df['clean_text'] = df['clean_text'].astype(str).fillna('')

# Tokenize and pad sequences
tokenizer = tf.keras.preprocessing.text.Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])

# Using 'total_tokens' column for dynamic padding
max_sequence_length = df['total_tokens'].max()  # Adjust maxlen dynamically
padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_sequence_length)

# Prepare labels
labels = df[['is_analysis', 'is_evaluation', 'is_synthesis']].values

# Create a stratification column based on the target labels
df['stratify_col'] = df[['is_analysis', 'is_evaluation',
                         'is_synthesis']].astype(str).agg(''.join, axis=1)

# Initialize DatasetSplitter with the new stratification column
splitter = DatasetSplitter(df, 'stratify_col')
folds = splitter.split_dataset()

# Define the get_embedding_matrix function


def get_embedding_matrix(
    word_index,
        embedding_dim=200,
        glove_file_path=os.path.join(DEFAULT_DATA_FOLDER, 'glove.6B', 'glove.6B.200d.txt')):
    """
    Returns the embedding matrix.
    """
    # Initialize the embedding matrix with zeros
    embedding_mat = np.zeros((len(word_index) + 1, embedding_dim))
    embedding_index = {}

    # Load the GloVe embeddings
    if not os.path.exists(glove_file_path):
        raise FileNotFoundError(
            f"{glove_file_path} not found. Please run download_gloves.py")

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
            embedding_mat[i] = embedding_vector

    return embedding_mat


# Create embedding matrix
embedding_matrix = get_embedding_matrix(tokenizer.word_index)

# Define hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
EPOCHS = 40
GRU_SIZE = 256
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 0

# Compile and train the model
model = PostPurposeGRU(len(tokenizer.word_index) + 1, embedding_matrix,
                       GRU_SIZE, HIDDEN_SIZE, NUM_HIDDEN_LAYERS).get_model()
optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=5, restore_best_weights=True)

# Cross-Validation Training
TEST_INDICES = None

for train_indices, val_indices, TEST_INDICES in folds:
    X_train_fold = padded_sequences[train_indices]
    X_val_fold = padded_sequences[val_indices]
    y_train_fold = labels[train_indices]
    y_val_fold = labels[val_indices]

    model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
              batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping])

# Final Evaluation on Test Set
test_loss, test_accuracy = model.evaluate(
    padded_sequences[TEST_INDICES], labels[TEST_INDICES])
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Save the model
model.save(os.path.join(DEFAULT_DATA_FOLDER, 'models', 'post_purpose_gru.h5'))
