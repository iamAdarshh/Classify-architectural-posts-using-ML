import pytest
import pandas as pd
from src.data.dataset_split import DatasetSplitter

@pytest.fixture
def sample_dataframe():
    data = {
        'Id': range(1, 101),
        'Text': [f'Title{i}' for i in range(1, 101)],
        'isArchitectural': [True, False] * 50,
        'isAnalysis': [False, True] * 50,
        'isFeature': [True, True, False, False] * 25,
        'isEvaluation': [False, False, True, True] * 25
    }
    df = pd.DataFrame(data)
    return df

def test_dataset_splitter_initialization(sample_dataframe):
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    assert splitter.df.equals(sample_dataframe)
    assert splitter.column == 'isArchitectural'
    assert splitter.train_percentage == 0.8
    assert splitter.val_percentage == 0.1
    assert splitter.n_splits == 10

def test_stratified_split(sample_dataframe):
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    splitter.stratified_split()
    assert len(splitter.indices) == 10
    for indices in splitter.indices:
        assert len(indices) > 0

def test_get_folds(sample_dataframe):
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    splitter.stratified_split()
    folds = splitter.get_folds()
    assert len(folds) == 10
    for train_indices, val_indices, test_indices in folds:
        assert len(train_indices) > 0
        assert len(val_indices) > 0
        assert len(test_indices) > 0

def test_split_dataset(sample_dataframe):
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    folds = splitter.split_dataset()
    assert len(folds) == 10
    for train_indices, val_indices, test_indices in folds:
        assert len(train_indices) > 0
        assert len(val_indices) > 0
        assert len(test_indices) > 0
        # Check proportions
        total_samples = len(sample_dataframe)
        train_set = sample_dataframe.iloc[train_indices]
        val_set = sample_dataframe.iloc[val_indices]
        test_set = sample_dataframe.iloc[test_indices]

        train_percentage = len(train_indices) / total_samples
        val_percentage = len(val_indices) / total_samples
        test_percentage = len(test_indices) / total_samples

        assert 0.70 <= train_percentage <= 0.85
        assert 0.05 <= val_percentage <= 0.15
        assert 0.05 <= test_percentage <= 0.15

        # Check stratification
        train_dist = train_set['isArchitectural'].value_counts(normalize=True).round(1).tolist()
        val_dist = val_set['isArchitectural'].value_counts(normalize=True).round(1).tolist()
        test_dist = test_set['isArchitectural'].value_counts(normalize=True).round(1).tolist()

        assert train_dist == [0.5, 0.5]
        assert val_dist in ([0.5, 0.5], [0.6, 0.4], [0.4, 0.6])
        assert test_dist == [0.5, 0.5]

if __name__ == "__main__":
    pytest.main()
