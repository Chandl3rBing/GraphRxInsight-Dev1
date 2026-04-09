import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score

print("Loading dataset...")

X = np.load("DATASETS/processed/X_hard_chunk_0_scaled.npy")
y = np.load("DATASETS/processed/y_hard_chunk_0.npy")

print("Dataset shape:", X.shape)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)

param_grid = {
    "n_estimators": [200, 400, 600, 800],
    "max_depth": [20, 30, 40, 50, None],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", "log2", 0.3, 0.5, 0.7],
    "class_weight": [None, "balanced"]
}

base_rf = RandomForestClassifier(
    n_jobs=-1,
    random_state=42,
    verbose=1
)

grid = GridSearchCV(
    estimator=base_rf,
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
    verbose=2,
    refit=True
)

print("\nStarting RF hyperparameter search...\n")

grid.fit(X_train, y_train)

best_rf = grid.best_estimator_
print("\nBest parameters:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)

print("\nEvaluating best RF on holdout test set...")
preds = best_rf.predict(X_test)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds)
print("Holdout Accuracy:", acc)
print("Holdout F1:", f1)

print("\nSaving best model...")
joblib.dump(best_rf, "models/rf_optimized.pkl")
print("Finished.")