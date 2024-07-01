import os
import xml.etree.ElementTree as ET

from src.config import DEFAULT_DATA_FOLDER

XML_PATH = os.path.join(DEFAULT_DATA_FOLDER, 'input', 'stackoverflow_questions.xml')  # Path to the XML file

def main():
    # Parse the XML file
    tree = ET.parse(XML_PATH)
    root = tree.getroot()

    # Initialize a counter for questions
    question_count = 0

    # Iterate through each question element
    for question in root.findall('.//question'):
        question_count += 1  # Increment the counter for each question

    # Print the total number of questions
    print(f'Total number of questions: {question_count}')

if __name__ == "__main__":
    main()
