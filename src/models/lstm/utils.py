import pandas as pd


def determine_labels(row: pd.Series) -> str:
    if row['is_evaluation']:
        return 'evaluation'
    elif row['is_analysis']:
        return 'analysis'
    elif row['is_synthesis']:
        return 'synthesis'
    else:
        return 'other'
