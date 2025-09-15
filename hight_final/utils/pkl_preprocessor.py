"""
Pickle Classification Preprocessor Module

A simple module to preprocess pickle files with 'Encrypted_data' and 'Class' columns for classification tasks.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle
import ast
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import ast



def load_and_preprocess_pickle(
    pickle_path, test_size=0.2, random_state=42, return_encoded=False, verbose=False
):
    """
    Load and preprocess a pickle file for classification.

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file containing 'Encrypted_data' and 'Class' columns
    test_size : float, default=0.2
        Proportion of dataset to include in test split
    random_state : int, default=42
        Random state for reproducibility
    return_encoded : bool, default=False
        If True, returns one-hot encoded labels. If False, returns original labels
    verbose : bool, default=False
        If True, prints information about the data

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test) - numpy arrays ready for classification
        If return_encoded=True, y_train and y_test will be one-hot encoded
    """

    # Load the pickle file
    df = pd.read_pickle(pickle_path)

    if verbose:
        print(f"Loaded pickle from: {pickle_path}")
        print(f"Original data shape: {df.shape}")

    # Validate required columns
    if "Encrypted_data" not in df.columns or "Class" not in df.columns:
        raise ValueError("DataFrame must contain 'Encrypted_data' and 'Class' columns")

    # Convert encrypted data column to numpy arrays
    def convert_data_to_array(data):
        """Convert encrypted data to numpy array"""
        if isinstance(data, str):
            # Handle string representations
            if data.startswith("[") and data.endswith("]"):
                clean_string = data[1:-1]
                data_list = [int(x.strip()) for x in clean_string.split(",")]
            else:
                data_list = [int(bit) for bit in data]
            return np.array(data_list, dtype=int)
        elif isinstance(data, (list, np.ndarray)):
            # Direct conversion for lists/arrays (handles existing nparrays)
            return np.array(data, dtype=int)
        else:
            return np.array(data, dtype=int)

    # Convert encrypted data column to arrays
    X = np.array([convert_data_to_array(data) for data in df["Encrypted_data"]])
    y = df["Class"].values

    if verbose:
        print(f"Feature matrix shape: {X.shape}")
        print(f"Number of unique classes: {len(np.unique(y))}")

    # Shuffle the data
    X_shuffled, y_shuffled = shuffle(X, y, random_state=random_state)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X_shuffled,
        y_shuffled,
        test_size=test_size,
        random_state=random_state,
        stratify=y_shuffled,  # Ensures balanced split across classes
    )

    if verbose:
        print(f"Train set shape: {X_train.shape}")
        print(f"Test set shape: {X_test.shape}")

    # One-hot encode if requested
    if return_encoded:
        encoder = OneHotEncoder(sparse_output=False)
        y_train = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test = encoder.transform(y_test.reshape(-1, 1))

        if verbose:
            print(f"One-hot encoded labels shape: {y_train.shape}")

    return X_train, X_test, y_train, y_test


def load_pickle_quick(pickle_path, **kwargs):
    """
    Quick loader with default parameters - just pass the pickle path.

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file
    **kwargs : dict
        Additional keyword arguments to pass to load_and_preprocess_pickle

    Returns:
    --------
    tuple
        (X_train, X_test, y_train, y_test)
    """
    return load_and_preprocess_pickle(pickle_path, **kwargs)


def get_class_info(pickle_path):
    """
    Get basic information about classes in the dataset.

    Parameters:
    -----------
    pickle_path : str
        Path to the pickle file

    Returns:
    --------
    dict
        Dictionary containing class information
    """
    df = pd.read_pickle(pickle_path)
    classes = df["Class"].unique()
    class_counts = df["Class"].value_counts()

    return {
        "classes": classes,
        "num_classes": len(classes),
        "class_counts": class_counts,
        "total_samples": len(df),
    }


def _convert_data_to_array(data):
    """Convert encrypted data to numpy array"""
    if isinstance(data, str):
        # Handle string representations
        if data.startswith("[") and data.endswith("]"):
            clean_string = data[1:-1]
            data_list = [int(x.strip()) for x in clean_string.split(",")]
        else:
            data_list = [int(bit) for bit in data]
        return np.array(data_list, dtype=int)
    elif isinstance(data, (list, np.ndarray)):
        # Direct conversion for lists/arrays
        return np.array(data, dtype=int)
    else:
        return np.array(data, dtype=int)


class PickleBatchLoader:
    """
    Loads a large pickle file and prepares it for batch-wise training.

    This loader separates a hold-out test set and provides a generator
    to iterate through the training data in smaller batches, reducing memory usage.
    """
    def __init__(self, pickle_path, batch_size=5000, test_size=0.2, random_state=42):
        """
        Initializes the loader, loads data, and creates train/test splits.

        Args:
            pickle_path (str): Path to the pickle file.
            batch_size (int): The number of samples per batch.
            test_size (float): The proportion of the dataset to hold out for testing.
            random_state (int): Seed for reproducibility.
        """
        print(f"Loading data from {pickle_path}...")
        df = pd.read_pickle(pickle_path)
        df = shuffle(df, random_state=random_state)

        # Separate features (X) and labels (y)
        X_raw = df["Encrypted_data"]
        y_raw = df["Class"]

        # Create a stratified split for a representative test set
        X_train_raw, X_test_raw, y_train_raw, y_test_raw = train_test_split(
            X_raw,
            y_raw,
            test_size=test_size,
            random_state=random_state,
            stratify=y_raw,
        )

        self.X_train_raw = X_train_raw
        self.y_train_raw = y_train_raw
        self.batch_size = batch_size
        self.n_samples = len(self.X_train_raw)
        self.label_encoder = LabelEncoder()

        print("Preprocessing hold-out test set...")
        # Preprocess and store the test set
        self.X_test = np.array([_convert_data_to_array(data) for data in X_test_raw])
        self.y_test_encoded = self.label_encoder.fit_transform(y_test_raw)

        # Fit the label encoder on the full training data labels to ensure consistency
        self.label_encoder.fit(self.y_train_raw)

        print("Loader initialized. Ready to generate batches.")

    def _preprocess_batch(self, X_batch_raw, y_batch_raw):
        """Preprocesses a single batch of data."""
        X_batch = np.array([_convert_data_to_array(data) for data in X_batch_raw])
        y_batch_encoded = self.label_encoder.transform(y_batch_raw)
        return X_batch, y_batch_encoded

    def batch_generator(self):
        """A generator that yields preprocessed training batches."""
        for i in range(0, self.n_samples, self.batch_size):
            X_batch_raw = self.X_train_raw.iloc[i : i + self.batch_size]
            y_batch_raw = self.y_train_raw.iloc[i : i + self.batch_size]

            if X_batch_raw.empty:
                continue

            yield self._preprocess_batch(X_batch_raw, y_batch_raw)

    def get_test_set(self):
        """Returns the preprocessed, held-out test set."""
        return self.X_test, self.y_test_encoded

    def __len__(self):
        """Returns the total number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

# Example usage when running as script
if __name__ == "__main__":
    # Example usage
    pickle_path = "your_file.pkl"  # Replace with your actual path

    # Basic usage
    X_train, X_test, y_train, y_test = load_pickle_quick(pickle_path)

    print("Data loaded successfully!")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # Get class information
    class_info = get_class_info(pickle_path)
    print(f"\nClass information: {class_info}")

    # Usage with one-hot encoding
    X_train_enc, X_test_enc, y_train_enc, y_test_enc = load_pickle_quick(
        pickle_path, return_encoded=True
    )
    print(f"\nWith one-hot encoding:")
    print(f"y_train_encoded shape: {y_train_enc.shape}")
    print(f"y_test_encoded shape: {y_test_enc.shape}")
