import os
import requests
import zipfile

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import DEFAULT_DATA_FOLDER

def download_glove(destination_folder, glove_file_url='https://nlp.stanford.edu/data/glove.6B.zip'):
    # Ensure the destination folder exists
    if not os.path.exists(destination_folder):
        os.makedirs(destination_folder)

    # Download the GloVe zip file
    glove_zip_path = os.path.join(destination_folder, 'glove.6B.zip')
    if not os.path.exists(glove_zip_path):
        print(f'Downloading GloVe embeddings from {glove_file_url}...')
        response = requests.get(glove_file_url, stream=True)
        with open(glove_zip_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    file.write(chunk)
        print('Download complete.')
    else:
        print('GloVe zip file already exists.')

    # Extract the GloVe zip file
    print('Extracting GloVe embeddings...')
    with zipfile.ZipFile(glove_zip_path, 'r') as zip_ref:
        zip_ref.extractall(destination_folder)
    print('Extraction complete.')

    # Remove the zip file to save space
    os.remove(glove_zip_path)
    print('Removed the zip file.')

# Specify the destination folder for GloVe embeddings
destination_folder = os.path.join(DEFAULT_DATA_FOLDER, 'glove.6B')

# Download and extract GloVe embeddings
download_glove(destination_folder)
