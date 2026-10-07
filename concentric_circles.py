import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
import numpy as np
learning_rate = 0.01
class Model(torch.nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_neurons):
        super().__init__()
        self.layer1 = torch.nn.Linear(n_inputs, hidden_neurons)
        self.layer2 = torch.nn.Linear(hidden_neurons, n_outputs) #Create 2 layers of linear regeression
    def forward(self, X):
        Z1 = self.layer1(X)
        A1 = torch.tanh(Z1)
        Z2 = self.layer2(A1)
        return Z2
X, y = make_circles(
    n_samples = 500,
    noise = 0.15,
    random_state = 42
    )
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42)
X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
y_train = torch.tensor(y_train).float()
y_test = torch.tensor(y_test).float()
loss_function = torch.nn.BCEWithLogitsLoss()
model = Model(2, 1, 16)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
for epoch in range(5000):
    y_hat = model(X_train)
    loss = loss_function(y_hat, y_train)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 500 == 0:
        print(epoch, loss.item())
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= 0.5
    correct = (y_test == predictions)
    accuracy = correct.float().mean()
print("Test accuracy:", accuracy.item())
    
