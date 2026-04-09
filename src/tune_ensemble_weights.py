import numpy as np
import torch
import joblib
from sklearn.metrics import accuracy_score


X=np.load("DATASETS/processed/X_chunk_9.npy")
y=np.load("DATASETS/processed/y_chunk_9.npy")


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


nn=DDIModel(X.shape[1])
nn.load_state_dict(
torch.load("models/nn_3284_model.pth",map_location="cpu")
)
nn.eval()

rf=joblib.load("models/rf_model.pkl")
xgb=joblib.load("models/xgb_model.pkl")


Xt=torch.tensor(X,dtype=torch.float32)

nn_pred=nn(Xt).detach().numpy().flatten()
rf_pred=rf.predict_proba(X)[:,1]
xgb_pred=xgb.predict_proba(X)[:,1]


best_acc=0
best_weights=None


for a in np.arange(0.1,0.8,0.1):
 for b in np.arange(0.1,0.8,0.1):
  c=1-a-b

  if c<=0:
   continue

  pred=a*nn_pred+b*xgb_pred+c*rf_pred

  binary=(pred>0.5).astype(int)

  acc=accuracy_score(y,binary)

  if acc>best_acc:

   best_acc=acc
   best_weights=(a,b,c)


print("Best Accuracy:",best_acc)

print("Best Weights (NN,XGB,RF):",best_weights)