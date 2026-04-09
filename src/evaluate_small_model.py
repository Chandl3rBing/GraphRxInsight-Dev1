import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score


print("Loading dataset...")

# Load dataset
X = np.load("DATASETS/processed/X_chunk_0.npy")
y = np.load("DATASETS/processed/y_chunk_0.npy")

print("Dataset Shape:", X.shape)


# Convert to tensor
X = torch.tensor(X, dtype=torch.float32)


# Detect input dimension automatically
input_dim = X.shape[1]

print("Input Dimension:", input_dim)


# Model Definition (must match training model)
class DDIModel(torch.nn.Module):

    def __init__(self,input_dim):

        super().__init__()

        self.fc1 = torch.nn.Linear(input_dim,512)
        self.fc2 = torch.nn.Linear(512,256)
        self.fc3 = torch.nn.Linear(256,1)

        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()


    def forward(self,x):

        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))

        return x



print("\nLoading model...")

model = DDIModel(input_dim)

model.load_state_dict(
torch.load(
"models/small_model_100k.pth",
map_location="cpu"
)
)

model.eval()


print("Evaluating model...\n")


# Predict
with torch.no_grad():

    predictions = model(X).numpy()


predictions = (predictions > 0.5).astype(int)


# Results
accuracy = accuracy_score(y, predictions)

f1 = f1_score(y, predictions)


print("RESULTS")
print("Accuracy:", accuracy)
print("F1 Score:", f1)