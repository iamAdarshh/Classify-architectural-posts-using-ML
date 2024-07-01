# pylint: disable=no-member

"""
This module includes the training of the GRU model.
"""

import os
import sys

# Add the src directory to the system path
sys.path.append(os.path.abspath(os.path.join(
    os.path.dirname(__file__), '..', '..', '..')))

import tensorflow as tf
import pandas as pd
import numpy as np
from gensim.models import KeyedVectors
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.initializers import Constant
import matplotlib.pyplot as plt
from src.config import DEFAULT_DATA_FOLDER
from src.data.dataset_split import DatasetSplitter
from src.models.gru.post_purpose import PostPurposeGRU

# Define the determine_labels function
def determine_labels(row: pd.Series) -> str:
    """
    Determine the label for a given row based on its content.

    Parameters:
    row (pd.Series): A pandas Series representing a row of data.

    Returns:
    str: The label for the row ('evaluation', 'analysis', 'synthesis', or 'other').
    """
    if row['is_evaluation']:
        return 'evaluation'
    if row['is_analysis']:
        return 'analysis'
    if row['is_synthesis']:
        return 'synthesis'
    return 'other'

# Load and prepare your dataset
df = pd.read_excel(os.path.join(DEFAULT_DATA_FOLDER, 'output',
                                'combined_posts_results.xlsx'), engine='openpyxl')

# Ensure clean_text is string and handle missing values
df['clean_text'] = df['clean_text'].astype(str).fillna('')

# Label encoding
df['category'] = df.apply(determine_labels, axis=1)
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])
num_classes = len(label_encoder.classes_)  # Correctly determine the number of classes

# Tokenize and pad sequences
max_sequence_length = 250  # Use a fixed sequence length for padding
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['clean_text'])
sequences = tokenizer.texts_to_sequences(df['clean_text'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

# Prepare labels
labels = tf.keras.utils.to_categorical(df['category'], num_classes=num_classes)

# Create a stratification column based on the target labels
df['stratify_col'] = df[['is_analysis', 'is_evaluation', 'is_synthesis']].astype(str).agg(''.join, axis=1)

# Initialize DatasetSplitter with the new stratification column
splitter = DatasetSplitter(df, 'stratify_col')
folds = splitter.split_dataset()

# Define the get_embedding_matrix function
def get_embedding_matrix(word_index, embedding_dim=200, embedding_file_path=f'{DEFAULT_DATA_FOLDER}/word-embedding/SO_vectors_200.bin'):
    """Returns the embedding matrix."""
    # Load pre-trained Word2Vec embeddings
    word2vec_model = KeyedVectors.load_word2vec_format(embedding_file_path, binary=True)

    # Initialize the embedding matrix with zeros
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    # Map the Word2Vec embeddings to the word index
    for word, i in word_index.items():
        if word in word2vec_model:
            embedding_matrix[i] = word2vec_model[word]

    return embedding_matrix

# Create embedding matrix
embedding_matrix = get_embedding_matrix(tokenizer.word_index)

# Define hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 40
GRU_SIZE = 256
HIDDEN_SIZE = 128
NUM_HIDDEN_LAYERS = 1  # Added an additional hidden layer for experimentation

# Compile and train the model
model = PostPurposeGRU(len(tokenizer.word_index) + 1, embedding_matrix,
                       GRU_SIZE, HIDDEN_SIZE, NUM_HIDDEN_LAYERS).get_model()
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))  # Ensure the output layer has correct number of units

optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
loss = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.0001)

# Print the hyperparameters
print("Hyperparameters:")
print(f"BATCH_SIZE: {BATCH_SIZE}")
print(f"LEARNING_RATE: {LEARNING_RATE}")
print(f"EPOCHS: {EPOCHS}")
print(f"GRU_SIZE: {GRU_SIZE}")
print(f"HIDDEN_SIZE: {HIDDEN_SIZE}")
print(f"NUM_HIDDEN_LAYERS: {NUM_HIDDEN_LAYERS}")

# Cross-Validation Training
accuracies = []
fold_no = 1

# Lists to store loss values for all folds
train_loss_per_fold = []
val_loss_per_fold = []

for train_indices, val_indices, test_indices in folds:
    print(f'Training on fold {fold_no}...')
    
    X_train_fold = padded_sequences[train_indices]
    X_val_fold = padded_sequences[val_indices]
    y_train_fold = labels[train_indices]
    y_val_fold = labels[val_indices]

    history = model.fit(X_train_fold, y_train_fold, validation_data=(X_val_fold, y_val_fold),
                        batch_size=BATCH_SIZE, epochs=EPOCHS, callbacks=[early_stopping, reduce_lr], verbose=2)

    # Store the training and validation loss
    train_loss_per_fold.append(history.history['loss'])
    val_loss_per_fold.append(history.history['val_loss'])

    # Evaluate the model
    loss, accuracy = model.evaluate(padded_sequences[test_indices], labels[test_indices], verbose=2)
    print(f'Test Accuracy for fold {fold_no}: {accuracy}')
    accuracies.append(accuracy)
    
    fold_no += 1

# Calculate the average accuracy across all folds
average_accuracy = np.mean(accuracies)
print(f'Average Test Accuracy: {average_accuracy}')

# Save the model
model.save(os.path.join(DEFAULT_DATA_FOLDER, 'models', 'post_purpose_gru.h5'))

# Flatten the lists of loss values for plotting
train_loss_all_folds = [loss for fold_loss in train_loss_per_fold for loss in fold_loss]
val_loss_all_folds = [loss for fold_loss in val_loss_per_fold for loss in fold_loss]

# Plot training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(train_loss_all_folds, label='Training Loss')
plt.plot(val_loss_all_folds, label='Validation Loss')
plt.title('Training and Validation Loss Across All Folds')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

# Save the plot to a file instead of displaying it
plot_path = os.path.join(DEFAULT_DATA_FOLDER, 'rnu_training_validation_loss.png')
plt.savefig(plot_path)
plt.close()
