from src.config import DEFAULT_DATA_FOLDER
from src.models.lstm.utils import determine_labels
from src.data.utils import save_as_excel

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from gensim.models import KeyedVectors
from tensorflow.keras.initializers import Constant
from sklearn.metrics import precision_score, recall_score, f1_score

import matplotlib.pyplot as plt

# load datasets
# Load the dataset
DATASET_PATH = f"{DEFAULT_DATA_FOLDER}/output/combined_posts_results.xlsx"
dataset = pd.read_excel(DATASET_PATH)

# Load pre-trained Word2Vec embeddings
word2vec_path = f'{DEFAULT_DATA_FOLDER}/word-embedding/SO_vectors_200.bin'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_path,binary=True)

# Prepare the dataset
dataset['category'] = dataset.apply(determine_labels, axis=1)
dataset['clean_text'] = dataset['clean_text'].fillna('').astype(str)


label_encoder = LabelEncoder()
dataset['category'] = label_encoder.fit_transform(dataset['category'])
num_classes = len(label_encoder.classes_)

# Tokenize and pad sequences
max_num_words = dataset['clean_text'].str.len().max()
max_sequence_length = 250
tokenizer = Tokenizer(num_words=max_num_words)
tokenizer.fit_on_texts(dataset['clean_text'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(dataset['clean_text'])
padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
labels_categorical = to_categorical(dataset['category'], num_classes=num_classes)

# hyper-parameters: these are calculated using optimize parameters notebook
embedding_dim = 200
lstm_units = 100
dropout_rate = 0.2
batch_size = 16
epochs = 20

kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
fold_no = 1

fold_accuracies = []
fold_precisions = []
fold_recalls = []
fold_f1_scores = []

# To store the results
results = []

for train_index, test_index in kf.split(padded_sequences, dataset['category']):
    print(
        f'Training on fold {fold_no} with params: embedding_dim={embedding_dim}, lstm_units={lstm_units}, dropout_rate={dropout_rate}, batch_size={batch_size}')

    X_train, X_test = padded_sequences[train_index], padded_sequences[test_index]
    y_train, y_test = labels_categorical[train_index], labels_categorical[test_index]

    # Build the LSTM model with pre-trained embeddings
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model:
            embedding_matrix[i] = word2vec_model[word]

    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=embedding_dim,
                        embeddings_initializer=Constant(embedding_matrix),
                        trainable=False))
    model.add(SpatialDropout1D(dropout_rate))
    model.add(Bidirectional(LSTM(lstm_units, dropout=dropout_rate, recurrent_dropout=dropout_rate)))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=3, mode='min', restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1,
                        callbacks=[early_stopping], verbose=2)

    # Evaluate the model
    y_pred = model.predict(X_test, verbose=2)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    accuracy = np.mean(y_pred_classes == y_true)
    precision = precision_score(y_true, y_pred_classes, average='weighted')
    recall = recall_score(y_true, y_pred_classes, average='weighted')
    f1 = f1_score(y_true, y_pred_classes, average='weighted')

    print(f'Test Accuracy for fold {fold_no}: {accuracy}')
    print(f'Test Precision for fold {fold_no}: {precision}')
    print(f'Test Recall for fold {fold_no}: {recall}')
    print(f'Test F1 Score for fold {fold_no}: {f1}')

    fold_accuracies.append(accuracy)
    fold_precisions.append(precision)
    fold_recalls.append(recall)
    fold_f1_scores.append(f1)

    # Store the results
    fold_results = pd.DataFrame({
        'fold': fold_no,
        'true': y_true,
        'predicted': y_pred_classes,
        'f1': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
    })
    results.append(fold_results)

    fold_no += 1

    save_as_excel(fold_results, f"{DEFAULT_DATA_FOLDER}/validation_results/fold{fold_no}.xlsx")

print("Fold accuracies: ", fold_accuracies)
print("Fold precisions: ", fold_precisions)
print("Fold recalls: ", fold_recalls)
print("Fold f1 scores: ", fold_f1_scores)

# Calculate average metrics for current hyperparameters
average_accuracy = np.mean(fold_accuracies)
average_precision = np.mean(fold_precisions)
average_recall = np.mean(fold_recalls)
average_f1_score = np.mean(fold_f1_scores)

print(f'Average Test Accuracy for config {embedding_dim}, {lstm_units}, {dropout_rate}, {batch_size}: {average_accuracy}')
print(f'Average Test Precision for config {embedding_dim}, {lstm_units}, {dropout_rate}, {batch_size}: {average_precision}')
print(f'Average Test Recall for config {embedding_dim}, {lstm_units}, {dropout_rate}, {batch_size}: {average_recall}')
print(f'Average Test F1 Score for config {embedding_dim}, {lstm_units}, {dropout_rate}, {batch_size}: {average_f1_score}')

# Plot the training history of the best model (for demonstration, assuming the last trained model is the best)
plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

plt.savefig(f"{DEFAULT_DATA_FOLDER}/charts/Training_loss_vs_loss.png")

import os

# Ensure the model directory exists
model_dir = os.path.join(DEFAULT_DATA_FOLDER, "models")
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

model.save(f"{DEFAULT_DATA_FOLDER}/models/lstm.keras")