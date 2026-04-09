import os

import numpy as np


DATA_PATH_X = "DATASETS/processed/X_dynamic.npy"
DATA_PATH_Y = "DATASETS/processed/y_dynamic.npy"


def update_dataset(X_new, y_new):
    X_new = np.asarray(X_new, dtype=np.float32)
    y_new = np.asarray(y_new, dtype=np.float32).reshape(-1)

    if X_new.ndim != 2:
        raise ValueError("X_new must be a 2D feature matrix.")

    if len(X_new) != len(y_new):
        raise ValueError("X_new and y_new must contain the same number of samples.")

    if os.path.exists(DATA_PATH_X) and os.path.exists(DATA_PATH_Y):
        X_old = np.load(DATA_PATH_X).astype(np.float32)
        y_old = np.load(DATA_PATH_Y).astype(np.float32)

        if X_old.ndim != 2 or y_old.ndim != 1 or len(X_old) != len(y_old):
            raise ValueError("Existing dynamic dataset files are malformed.")

        if len(X_old) > 0 and X_old.shape[1] != X_new.shape[1]:
            raise ValueError(
                f"Existing dynamic feature dimension {X_old.shape[1]} does not match new "
                f"feature dimension {X_new.shape[1]}."
            )

        if len(X_old) == 0:
            X = X_new
            y = y_new
        else:
            X = np.vstack([X_old, X_new])
            y = np.concatenate([y_old, y_new])
    else:
        X = X_new
        y = y_new

    np.save(DATA_PATH_X, X.astype(np.float32))
    np.save(DATA_PATH_Y, y.astype(np.float32))

    print("Dataset updated")
    print("Samples:", len(X))
