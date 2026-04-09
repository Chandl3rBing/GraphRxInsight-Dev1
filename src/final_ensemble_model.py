import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score,f1_score


print("Loading Dataset")

X=np.load(
"DATASETS/processed/X_chunk_0_scaled.npy"
)

y=np.load(
"DATASETS/processed/y_chunk_0.npy"
)



# ---------- LOAD MODELS ----------

print("Loading Neural Model")

class Model(torch.nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1=torch.nn.Linear(input_dim,512)
        self.fc2=torch.nn.Linear(512,256)
        self.out=torch.nn.Linear(256,1)


    def forward(self,x):

        x=torch.relu(self.fc1(x))
        x=torch.relu(self.fc2(x))
        x=torch.sigmoid(self.out(x))

        return x


nn_model=Model(X.shape[1])

nn_model.load_state_dict(
torch.load(
"models/high_accuracy_model_v2.pth",
map_location="cpu"
)
)

nn_model.eval()


print("Loading RF Model")

rf=joblib.load(
"models/rf_model.pkl"
)


print("Loading XGB Model")

xgb=joblib.load(
"models/xgb_model.pkl"
)



# ---------- PREDICTIONS ----------

print("Predicting")

nn_pred=nn_model(
torch.tensor(X,dtype=torch.float32)
).detach().numpy().flatten()


rf_pred=rf.predict_proba(X)[:,1]

xgb_pred=xgb.predict_proba(X)[:,1]


# ---------- ENSEMBLE ----------

final_pred={.1*nn_pred+
0.4*xgb_pred+
0.5*rf_pred}


final_binary=(final_pred>0.5).astype(int)


# ---------- RESULTS ----------

acc=accuracy_score(y,final_binary)
f1=f1_score(y,final_binary)

print("\nFINAL RESULTS")
print("Accuracy:",acc)
print("F1 Score:",f1)