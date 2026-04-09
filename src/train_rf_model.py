import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier

print("Training Random Forest (3284 features)")

chunks=[3,4,5]

X_all=[]
y_all=[]

for i in chunks:

    print("Loading chunk:",i)

    X=np.load(f"DATASETS/processed/X_chunk_{i}.npy")
    y=np.load(f"DATASETS/processed/y_chunk_{i}.npy")

    print("Shape:",X.shape)

    X_all.append(X)
    y_all.append(y)

X=np.vstack(X_all)
y=np.concatenate(y_all)

print("\nFinal dataset:",X.shape)


model=RandomForestClassifier(

    n_estimators=200,
    max_depth=25,
    n_jobs=-1,
    random_state=42

)

print("\nTraining...")

model.fit(X,y)

joblib.dump(model,"models/rf_model.pkl")

print("\nRandom Forest Finished")