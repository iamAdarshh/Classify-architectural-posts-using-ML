import lxml.etree as ET
import pandas as pd


def parse_posts(path):
    context = ET.iterparse(path, events=('end',), tag='row')
    questions = []
    answers = []

    for event, elem in context:
        if elem.tag == 'row' and elem.attrib:
            attrib_dict = dict(elem.attrib)  # Convert to regular dictionary
            post_type = attrib_dict.get('PostTypeId')
            if post_type == '1':
                questions.append(attrib_dict)
            elif post_type == '2' and 'ParentId' in attrib_dict:
                answers.append(attrib_dict)

        elem.clear()
        while elem.getprevious() is not None:
            del elem.getparent()[0]

    return len(questions), len(answers)


if __name__ == '__main__':
    # Specify the path to your Posts.xml file
    file_path = '/scratch/hpc-prf-dssecs/Posts.xml'

    questions_df, answers_df = parse_posts(file_path)

    # Display the DataFrames
    print("Total number of questions: ", questions_df, answers_df)