import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold


class DatasetSplitter:
    def __init__(self, df: pd.DataFrame, column: str, train_percentage: float = 0.8, val_percentage: float = 0.1, n_splits: int = 10):
        """
        Initializes the DatasetSplitter class.

        @param df: The input dataframe containing the dataset.
        @param column: The column name used for stratified splitting.
        @param train_percentage: The percentage of data used for training (default is 0.8).
        @param val_percentage: The percentage of data used for validation (default is 0.1).
        @param n_splits: The number of splits for k-fold cross-validation (default is 10).
        """
        self.df = df
        self.column = column
        self.train_percentage = train_percentage
        self.val_percentage = val_percentage
        self.n_splits = n_splits
        self.indices = []

    def stratified_split(self):
        """
        Divides the dataset into stratified k-folds.
        """
        X = np.zeros(len(self.df))
        y = self.df[self.column]

        skf = StratifiedKFold(n_splits=self.n_splits,
                              random_state=42, shuffle=True)
        for _, test_index in skf.split(X, y):
            self.indices.append(test_index)

    def get_folds(self):
        """
        Returns the indices for k-fold cross-validation.
        """
        folds = []
        for i in range(self.n_splits):
            test_indices = self.indices[i]
            remaining_indices = [self.indices[j]
                                 for j in range(self.n_splits) if j != i]
            remaining_indices = [
                idx for sublist in remaining_indices for idx in sublist]

            train_split = int(len(remaining_indices) * self.train_percentage)
            val_split = int(len(remaining_indices) * self.val_percentage)

            train_indices = remaining_indices[:train_split]
            val_indices = remaining_indices[train_split:train_split + val_split]

            folds.append((train_indices, val_indices, test_indices))

        return folds

    def split_dataset(self):
        """
        The main function that performs the dataset splitting and returns the k-folds.

        @return: A list of tuples, each containing train, validation, and test indices.
        """
        self.stratified_split()
        folds = self.get_folds()
        return folds


if __name__ == "__main__":
    pytest.main()
