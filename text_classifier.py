import torch
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import numpy as np
class Model(torch.nn.Module):
    def __init__(self, dim_inputs, dim_outputs, hidden_neurons):
        super().__init__()
        self.layer1 = torch.nn.Linear(dim_inputs, hidden_neurons)
        self.layer2 = torch.nn.Linear(hidden_neurons, dim_outputs)
    def forward(self, X):
        Z1 = self.layer1(X) #So we will first pass the randomly generated layer through our input X
        A1 = torch.tanh(Z1) # Add nonlinearity so the network can learn a curved decision boundary
        Z2 = self.layer2(A1)
        return Z2
learning_rate = 0.01
X, y = make_moons(
    n_samples = 500,
    noise = 0.15,
    random_state = 42)
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
    )
X_train = torch.tensor(X_train).float()
y_train = torch.tensor(y_train).float()
X_test = torch.tensor(X_test).float()
y_test = torch.tensor(y_test).float()
loss_function = torch.nn.BCEWithLogitsLoss()
model = Model(2, 1, 8)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
for epoch in range(5000):
    #Forward Pass done manually:
    y_hat = model(X_train)
    #compute loss
    loss = loss_function(y_hat, y_train)
    #Run backprop to get gradient values:
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    if epoch % 500 == 0:
        print(loss.item())
model.eval()

with torch.no_grad():
    logits = model(X_test)
    probabilities = torch.sigmoid(logits)
    predictions = probabilities >= 0.5
    correct = (y_test == predictions)
    accuracy = correct.float().mean()
print("Test accuracy:", accuracy.item())
