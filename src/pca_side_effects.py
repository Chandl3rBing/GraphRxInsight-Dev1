import pandas as pd
from sklearn.decomposition import PCA

print("Loading side effects features...")

df = pd.read_csv(
"DATASETS/processed/side_effect_features.csv"
)

drug_ids = df.iloc[:,0]

X = df.iloc[:,1:]


print("Original shape:",X.shape)


pca = PCA(n_components=200)

X_reduced = pca.fit_transform(X)


print("Reduced shape:",X_reduced.shape)


out = pd.DataFrame(X_reduced)

out.insert(0,"drug_id",drug_ids)


out.to_csv(
"DATASETS/processed/side_effects_pca.csv",
index=False
)


print("Saved PCA side effects")
