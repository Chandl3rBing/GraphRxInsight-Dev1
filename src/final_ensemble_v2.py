import numpy as np
import torch
import torch.nn as nn
import joblib

from sklearn.metrics import accuracy_score,f1_score


#########################################
# Load Dataset
#########################################

print("Loading dataset...")

X = np.load(
"DATASETS/processed/X_hard_chunk_0_scaled.npy"
)

y = np.load(
"DATASETS/processed/y_hard_chunk_0.npy"
)

print("Dataset Shape:",X.shape)


#########################################
# Neural Network Model
#########################################

class FinalDDIModel(nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1 = nn.Linear(input_dim,512)
        self.bn1 = nn.BatchNorm1d(512)

        self.fc2 = nn.Linear(512,256)
        self.bn2 = nn.BatchNorm1d(256)

        self.fc3 = nn.Linear(256,128)
        self.bn3 = nn.BatchNorm1d(128)

        self.out = nn.Linear(128,1)

        self.relu = nn.ReLU()


    def forward(self,x):

        x = self.relu(self.bn1(self.fc1(x)))
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.relu(self.bn3(self.fc3(x)))

        x = torch.sigmoid(self.out(x))

        return x



#########################################
# Load Neural Model
#########################################

print("Loading Neural Model...")

input_dim = X.shape[1]

nn_model = FinalDDIModel(input_dim)

nn_model.load_state_dict(
torch.load(
"models/final_hard_model.pth",
map_location="cpu"
)
)

nn_model.eval()



#########################################
# Load Random Forest
#########################################

print("Loading Random Forest Model...")

rf_model = joblib.load(
"models/rf_model.pkl"
)



#########################################
# Load XGBoost
#########################################

print("Loading XGBoost Model...")

xgb_model = joblib.load(
"models/xgb_model.pkl"
)



#########################################
# Predictions
#########################################

print("Running Predictions...")


X_tensor = torch.tensor(
X,
dtype=torch.float32
)


# Neural Network
with torch.no_grad():

    nn_preds = nn_model(X_tensor).numpy().flatten()


# Random Forest
rf_preds = rf_model.predict_proba(X)[:,1]


# XGBoost
xgb_preds = xgb_model.predict_proba(X)[:,1]



#########################################
# Ensemble Weights
#########################################

print("Combining Models...")


nn_weight = 0.2
rf_weight = 0.4
xgb_weight = 0.4


ensemble_probs = (

nn_weight * nn_preds +

rf_weight * rf_preds +

xgb_weight * xgb_preds

)


ensemble_preds = (
ensemble_probs > 0.5
).astype(int)



#########################################
# Evaluation
#########################################

accuracy = accuracy_score(
y,
ensemble_preds
)

f1 = f1_score(
y,
ensemble_preds
)



print("\nFINAL ENSEMBLE RESULTS\n")

print("Accuracy:",accuracy)

print("F1 Score:",f1)