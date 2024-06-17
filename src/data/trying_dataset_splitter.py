import pandas as pd
from dataset_split import DatasetSplitter

# Generate a larger example dataframe
data = {
    'Id': range(1, 101),
    'Text': [f'Title{i}' for i in range(1, 101)],
    'isArchitectural': [True, False] * 50,
    'isAnalysis': [False, True] * 50,
    'isFeature': [True, True, False, False] * 25,
    'isEvaluation': [False, False, True, True] * 25
}

df = pd.DataFrame(data)

# Initialize the DatasetSplitter
splitter = DatasetSplitter(df, column='isArchitectural', train_percentage=0.8, val_percentage=0.1, n_splits=10)

# Perform the dataset splitting
folds = splitter.split_dataset()

# Check the distribution and actual split percentages for the first fold
train_indices, val_indices, test_indices = folds[0]

train_set = df.iloc[train_indices]
val_set = df.iloc[val_indices]
test_set = df.iloc[test_indices]

total_samples = len(df)

print("Training set distribution:")
print(train_set['isArchitectural'].value_counts(normalize=True))
print(f"Training set percentage: {len(train_indices) / total_samples * 100:.2f}%")

print("\nValidation set distribution:")
print(val_set['isArchitectural'].value_counts(normalize=True))
print(f"Validation set percentage: {len(val_indices) / total_samples * 100:.2f}%")

print("\nTest set distribution:")
print(test_set['isArchitectural'].value_counts(normalize=True))
print(f"Test set percentage: {len(test_indices) / total_samples * 100:.2f}%")
