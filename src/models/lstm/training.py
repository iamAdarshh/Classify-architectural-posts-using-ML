import pandas as pd
from src.config import DEFAULT_DATA_FOLDER
from utils import determine_labels
import tensorflow as tf

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, SpatialDropout1D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import StratifiedKFold
from typing import Union, Tuple, List

DATASET_PATH = f"{DEFAULT_DATA_FOLDER}/output/combined_posts_results.xlsx"


if __name__ =='__main__':

    dataset = pd.read_excel(DATASET_PATH)

    # labels the columns
    dataset['category'] = dataset.apply(determine_labels, axis=1)
    dataset['clean_text'] = dataset['clean_text'].fillna('').astype(str)

    label_encoder: LabelEncoder = LabelEncoder()
    dataset['category'] = label_encoder.fit_transform(dataset['category'])
    num_classes: int = len(label_encoder.classes_)



    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(dataset['clean_text'], dataset['category'], test_size=0.2,
                                                        random_state=42)

    # Tokenize and pad sequences
    max_num_words: int = dataset['clean_text'].str.len().max()
    print("Max number of words: ", max_num_words)
    max_sequence_length: int = 250
    tokenizer: Tokenizer = Tokenizer(num_words=max_num_words)
    tokenizer.fit_on_texts(X_train)
    X_train_sequences = tokenizer.texts_to_sequences(X_train)
    X_test_sequences = tokenizer.texts_to_sequences(X_test)

    X_train_padded: np.ndarray = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
    X_test_padded: np.ndarray = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

    # Convert labels to categorical
    y_train_categorical: np.ndarray = to_categorical(y_train, num_classes=num_classes)
    y_test_categorical: np.ndarray = to_categorical(y_test, num_classes=num_classes)

    sequences = tokenizer.texts_to_sequences(dataset['clean_text'])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    labels_categorical: np.ndarray = to_categorical(dataset['category'], num_classes=num_classes)

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1
    accuracies: List[float] = []

    for train_index, test_index in skf.split(padded_sequences, dataset['category']):
        print(f'Training on fold {fold_no}...')

        # Split the data into training and testing sets
        X_train, X_test = padded_sequences[train_index], padded_sequences[test_index]
        y_train, y_test = labels_categorical[train_index], labels_categorical[test_index]

        # Build the LSTM model
        embedding_dim: int = 100

        model: Sequential = Sequential()
        model.add(Embedding(input_dim=max_num_words, output_dim=embedding_dim, input_length=max_sequence_length))
        model.add(SpatialDropout1D(0.2))
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
        model.add(Dense(num_classes, activation='softmax'))

        # Compile the model
        model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])

        # Train the model
        #model.fit(X_train, y_train, epochs=5, batch_size=64, validation_split=0.1, verbose=2)

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        # Train the model with early stopping
        history = model.fit(X_train, y_train, epochs=20, batch_size=64, validation_split=0.1,
                            callbacks=[early_stopping], verbose=2)

        # Evaluate the model
        loss, accuracy = model.evaluate(X_test, y_test, verbose=2)
        print(f'Test Accuracy for fold {fold_no}: {accuracy}')
        accuracies.append(accuracy)

        fold_no += 1

    # Calculate the average accuracy across all folds
    average_accuracy = np.mean(accuracies)
    print(f'Average Test Accuracy: {average_accuracy}')