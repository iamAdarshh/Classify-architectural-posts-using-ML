from src.config import DEFAULT_DATA_FOLDER
from src.data.preprocessing import preprocess_text

import os
import xml.etree.ElementTree as ET

from tensorflow.keras.models import load_model


# # Path to the XML file
# xml_file = 'path_to_your_xml_file.xml'
#
# # Function to parse XML and yield each question
# def parse_xml(xml_file):
#     context = ET.iterparse(xml_file, events=('start', 'end'))
#     _, root = next(context)  # Get the root element
#     for event, elem in context:
#         if event == 'end' and elem.tag == 'row':
#             yield elem.attrib
#             root.clear()  # Clear the element to free memory
#
# Function to predict question type
def predict_question_type(model, text):
    processed_text = preprocess_text(text)
    prediction = model.predict(processed_text)
    return prediction


# # Traverse through questions and predict their type
# for question in parse_xml(xml_file):
#     # question_text = question.get('Body', '')  # Assuming 'Body' contains the question text
#
#


if __name__ == '__main__':
    # Load the trained LSTM model
    model_folder = f'{DEFAULT_DATA_FOLDER}/models'
    model_path = os.path.join(model_folder, 'lstm.keras')
    model = load_model(model_path)

    question_text = "pro con oledb versus sqlclient context one system work net web application vbnet front end sql server backend variety reason lose time original designer decide use net oledb connection rather sqlclient connection year development particular system cusp cross line beta status one thing talk point move sqlclient connection aware best practice use way get fancier feature sql server use obviously advantage use one hidden gotchas know anyone point benchmark show relative speed hear sqlclient suppose faster never see number back thanks oledb generic ever move different database type future good chance ole driver change much code hand sql server native driver suppose faster say nicer parameter support parameter use name order personal experience never notice speed difference also could find anything back claim suspect performance advantage real would process million record could start measure notice make meaningful difference error message trouble old oledb app switch sqlclient desperation course still work well error message provide enough new information able fix problem"
    prediction = predict_question_type(model, question_text)

    # Print or process the prediction as needed
    # print(f"Question: {question_text[:50]}...")
    print(f"Prediction: {prediction}")