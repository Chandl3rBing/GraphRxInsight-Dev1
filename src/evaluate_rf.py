import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score,f1_score

print("Loading Test Data")

X=np.load("DATASETS/processed/X_chunk_9.npy")
y=np.load("DATASETS/processed/y_chunk_9.npy")

print("Shape:",X.shape)


class DDIModel(torch.nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=torch.nn.Linear(input_dim,512)
        self.fc2=torch.nn.Linear(512,256)
        self.fc3=torch.nn.Linear(256,1)

    def forward(self,x):

        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))

        return x


print("\nLoading Neural Model")

input_dim=X.shape[1]

nn_model=DDIModel(input_dim)

nn_model.load_state_dict(
torch.load(
"models/high_accuracy_model_v2.pth",
map_location="cpu"
)
)

nn_model.eval()


print("Loading Random Forest")

rf_model=joblib.load("models/rf_model.pkl")


print("Loading XGBoost")

xgb_model=joblib.load("models/xgb_model.pkl")


print("\nPredicting")


X_tensor=torch.tensor(X,dtype=torch.float32)

nn_pred=nn_model(X_tensor).detach().numpy().flatten()

rf_pred=rf_model.predict_proba(X)[:,1]

xgb_pred=xgb_model.predict_proba(X)[:,1]


final_pred=(
0.4*nn_pred+
0.4*xgb_pred+
0.2*rf_pred
)


final_binary=(final_pred>0.5).astype(int)


print("\nRESULTS")

print("Accuracy:",accuracy_score(y,final_binary))

print("F1 Score:",f1_score(y,final_binary))