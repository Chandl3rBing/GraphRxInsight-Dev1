import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn.metrics import accuracy_score


print("Loading dataset...")

X = np.load(
"DATASETS/processed/X_hard_chunk_0_scaled.npy"
)

y = np.load(
"DATASETS/processed/y_hard_chunk_0.npy"
)


#################################
# Neural Model
#################################

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

        x=self.relu(self.bn1(self.fc1(x)))
        x=self.relu(self.bn2(self.fc2(x)))
        x=self.relu(self.bn3(self.fc3(x)))

        x=torch.sigmoid(self.out(x))

        return x


#################################
# Load Models
#################################

input_dim=X.shape[1]

nn_model=FinalDDIModel(input_dim)

nn_model.load_state_dict(
torch.load(
"models/final_hard_model.pth",
map_location="cpu"
)
)

nn_model.eval()


rf_model=joblib.load(
"models/rf_model.pkl"
)

xgb_model=joblib.load(
"models/xgb_model.pkl"
)


#################################
# Predictions
#################################

print("Generating predictions...")

X_tensor=torch.tensor(
X,
dtype=torch.float32
)


with torch.no_grad():

    nn_preds=nn_model(X_tensor).numpy().flatten()


rf_preds=rf_model.predict_proba(X)[:,1]

xgb_preds=xgb_model.predict_proba(X)[:,1]


#################################
# Search Best Weights
#################################

print("Searching best weights...")

best_acc=0
best_weights=None


for nn_w in np.arange(0,1.1,0.1):

    for rf_w in np.arange(0,1.1,0.1):

        for xgb_w in np.arange(0,1.1,0.1):

            total=nn_w+rf_w+xgb_w

            if total==0:
                continue

            nn_c=nn_w/total
            rf_c=rf_w/total
            xgb_c=xgb_w/total


            preds=(

            nn_c*nn_preds+

            rf_c*rf_preds+

            xgb_c*xgb_preds

            )


            preds=(preds>0.5).astype(int)


            acc=accuracy_score(y,preds)


            if acc>best_acc:

                best_acc=acc

                best_weights=(nn_c,rf_c,xgb_c)


print("\nBest Accuracy:",best_acc)

print("Best Weights (NN,RF,XGB):",best_weights)