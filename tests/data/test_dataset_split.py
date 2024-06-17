"""
Tests for the DatasetSplitter class.
"""

import pytest
import pandas as pd
from src.data.dataset_split import DatasetSplitter


@pytest.fixture
def sample_dataframe():
    """
    Fixture that provides a sample dataframe for testing.
    """
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
    """Test the initialization of the DatasetSplitter class."""
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    assert splitter.df.equals(sample_dataframe)
    assert splitter.column == 'isArchitectural'
    assert splitter.train_percentage == 0.8
    assert splitter.val_percentage == 0.1
    assert splitter.n_splits == 10


def test_stratified_split(sample_dataframe):
    """Test the stratified splitting of the dataset."""
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    splitter.stratified_split()
    assert len(splitter.indices) == 10
    for indices in splitter.indices:
        assert len(indices) > 0


def test_get_folds(sample_dataframe):
    """Test the retrieval of stratified k-folds."""
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    splitter.stratified_split()
    folds = splitter.get_folds()
    assert len(folds) == 10
    for train_indices, val_indices, test_indices in folds:
        assert len(train_indices) > 0
        assert len(val_indices) > 0
        assert len(test_indices) > 0


def test_split_dataset(sample_dataframe):
    """Test the full dataset splitting process."""
    splitter = DatasetSplitter(sample_dataframe, column='isArchitectural')
    folds = splitter.split_dataset()
    assert len(folds) == 10
    for train_idx, val_idx, test_idx in folds:
        assert len(train_idx) > 0
        assert len(val_idx) > 0
        assert len(test_idx) > 0
        # Check proportions
        total_samples = len(sample_dataframe)
        train_set = sample_dataframe.iloc[train_idx]
        val_set = sample_dataframe.iloc[val_idx]
        test_set = sample_dataframe.iloc[test_idx]

        train_pct = len(train_idx) / total_samples
        val_pct = len(val_idx) / total_samples
        test_pct = len(test_idx) / total_samples

        assert 0.70 <= train_pct <= 0.85  # Allowing slight variance
        assert 0.05 <= val_pct <= 0.15  # Allowing slight variance
        assert 0.05 <= test_pct <= 0.15  # Allowing slight variance

        # Check stratification
        train_dist = train_set['isArchitectural'].value_counts(
            normalize=True).round(1).tolist()
        val_dist = val_set['isArchitectural'].value_counts(
            normalize=True).round(1).tolist()
        test_dist = test_set['isArchitectural'].value_counts(
            normalize=True).round(1).tolist()

        assert train_dist == [0.5, 0.5]
        # Allowing slight variance
        assert val_dist in ([0.5, 0.5], [0.6, 0.4], [0.4, 0.6])
        assert test_dist == [0.5, 0.5]


if __name__ == "__main__":
    pytest.main()
