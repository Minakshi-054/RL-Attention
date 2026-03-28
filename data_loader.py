import os
import numpy as np
from sklearn.utils.class_weight import compute_class_weight


def load_fold_data(fold_dir):
    train_data = np.load(os.path.join(fold_dir, "train.npz"))
    val_data = np.load(os.path.join(fold_dir, "val.npz"))
    test_data = np.load(os.path.join(fold_dir, "test.npz"))

    X_train = train_data["X"].astype(np.float32)
    y_train = train_data["y"].astype(np.float32)

    X_val = val_data["X"].astype(np.float32)
    y_val = val_data["y"].astype(np.float32)

    X_test = test_data["X"].astype(np.float32)
    y_test = test_data["y"].astype(np.float32)

    return X_train, y_train, X_val, y_val, X_test, y_test


def normalize_data(X_train, X_val, X_test):
    """
    Normalize using training data only.
    """
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, X_val, X_test, mean, std


def get_class_weights(y_train):
    classes = np.unique(y_train.astype(int))
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train.astype(int)
    )
    return {int(c): float(w) for c, w in zip(classes, weights)}
