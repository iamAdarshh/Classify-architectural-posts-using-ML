import os
import pickle
import xml.etree.ElementTree as ET
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

from src.config import DEFAULT_DATA_FOLDER
from src.data.preprocessing import preprocess_text

# Constants
MODEL_PATH = os.path.join(DEFAULT_DATA_FOLDER, 'models', 'lstm.keras')
TOKENIZER_PATH = os.path.join(DEFAULT_DATA_FOLDER, 'tokenizer.pickle')
CLASSES_PATH = os.path.join(DEFAULT_DATA_FOLDER, 'classes.npy')
XML_PATH = os.path.join(DEFAULT_DATA_FOLDER, 'input', 'stackoverflow_questions.xml')  # Path to the XML file
OUTPUT_DIR = os.path.join(DEFAULT_DATA_FOLDER, 'results')  # Directory for saving the results

# Ensure the output directory exists
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


def main():
    # Load the model
    model = load_model(MODEL_PATH)

    # Load the tokenizer
    with open(TOKENIZER_PATH, 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load label encoder classes
    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load(CLASSES_PATH, allow_pickle=True)

    # Parse the XML file
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    questions = []
    question_properties = []

    # first tryout with the posts
    # dataset = pd.read_excel(f"{DEFAULT_DATA_FOLDER}/processed/architectural_posts_details.xlsx")
    #
    # for index, question in dataset.iterrows():
    #     title = question['title'] if question['title'] is not None else ''
    #     body = question['body'] if question['body'] is not None else ''
    #     score = int(question['score']) if question['score'] is not None else 0
    #     answer_count = int(question['answer_count']) if question['answer_count'] is not None else 0
    #     tags = len(question['tags'].split(',')) if question['tags'] is not None else 0
    #     full_text = title + " " + body
    #     questions.append(full_text)
    #     question_properties.append({'score': score, 'answer_count': answer_count, 'tag_count': tags})

    for question in root.findall('.//question'):
        title = question.find('title').text if question.find('title') is not None else ''
        body = question.find('body').text if question.find('body') is not None else ''
        score = int(question.find('score').text) if question.find('score') is not None else 0
        answer_count = int(question.find('answerCount').text) if question.find('answerCount') is not None else 0
        tags = question.find('tags').text.split(',').count() if question.find('tags') is not None else 0
        full_text = title + " " + body
        questions.append(full_text)
        question_properties.append({'score': score, 'answer_count': answer_count, 'tag_count': tags})

    # Initialize counters and property collectors
    post_counts = defaultdict(int)
    post_properties = defaultdict(lambda: {'score': [], 'answer_count': [], 'tag_count': []})
    all_predictions = []

    for idx, question in enumerate(questions):
        # Preprocess the text
        clean_text = preprocess_text(question)

        # Tokenize and pad the input text
        max_sequence_length = 250  # Same as used during training
        sequences = tokenizer.texts_to_sequences([clean_text])
        padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

        # Make predictions
        predictions = model.predict(padded_sequences)
        predicted_classes = np.argmax(predictions, axis=1)

        # Decode the predicted labels
        predicted_labels = label_encoder.inverse_transform(predicted_classes)
        predicted_label = predicted_labels[0]

        # Update counters and properties
        post_counts[predicted_label] += 1
        post_properties[predicted_label]['score'].append(question_properties[idx]['score'])
        post_properties[predicted_label]['answer_count'].append(question_properties[idx]['answer_count'])
        post_properties[predicted_label]['tag_count'].append(question_properties[idx]['tag_count'])

        # Store the predictions
        all_predictions.append({
            'question': question,
            'score': question_properties[idx]['score'],
            'answer_count': question_properties[idx]['answer_count'],
            'tag_count': question_properties[idx]['tag_count'],
            'predicted_class': predicted_label
        })

    # Convert predictions to DataFrame and save to Excel
    predictions_df = pd.DataFrame(all_predictions)
    predictions_df.to_excel(os.path.join(OUTPUT_DIR, 'predictions.xlsx'), index=False)

    # Prepare to store statistics in DataFrame
    stats_data = []
    for label in label_encoder.classes_:
        if label in post_properties:
            scores = post_properties[label]['score']
            answer_counts = post_properties[label]['answer_count']
            tags = post_properties[label]['tag_count']

            stats_data.append({
                'Category': label,
                'Total Posts': post_counts[label],
                'Score - Avg': np.mean(scores),
                'Score - Min': np.min(scores),
                'Score - Max': np.max(scores),
                'Answer Count - Avg': np.mean(answer_counts),
                'Answer Count - Min': np.min(answer_counts),
                'Answer Count - Max': np.max(answer_counts),
                'Tag - Avg': np.mean(tags),
                'Tag - Min': np.min(tags),
                'Tag - Max': np.max(tags),
            })
        else:
            stats_data.append({
                'Category': label,
                'Total Posts': 0,
                'Score - Avg': 'N/A',
                'Score - Min': 'N/A',
                'Score - Max': 'N/A',
                'Answer Count - Avg': 'N/A',
                'Answer Count - Min': 'N/A',
                'Answer Count - Max': 'N/A',
                'Tag - Avg': 'N/A',
                'Tag - Min': 'N/A',
                'Tag - Max': 'N/A',
            })

    stats_df = pd.DataFrame(stats_data)
    stats_df.to_excel(os.path.join(OUTPUT_DIR, 'statistics.xlsx'), index=False)

    # Print the totals and statistics
    for label in label_encoder.classes_:
        if label in post_properties:
            scores = post_properties[label]['score']
            answer_counts = post_properties[label]['answer_count']

            print(f"\nStatistics for {label} posts:")
            print(f"Total number of posts: {post_counts[label]}")
            print(f"Score - Avg: {np.mean(scores):.2f}, Min: {np.min(scores)}, Max: {np.max(scores)}")
            print(f"Answer Count - Avg: {np.mean(answer_counts):.2f}, Min: {np.min(answer_counts)}, Max: {np.max(answer_counts)}")
        else:
            print(f"\nStatistics for {label} posts:")
            print(f"Total number of posts: 0")
            print(f"Score - Avg: N/A, Min: N/A, Max: N/A")
            print(f"Answer Count - Avg: N/A, Min: N/A, Max: N/A")


if __name__ == "__main__":
    main()
