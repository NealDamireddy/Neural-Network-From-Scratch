import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from mlp import TwoLayerMLP
np.random.seed(42)
X, y = make_moons(
    n_samples = 500,
    noise = 0.15,
    random_state = 42
    )
y = y.reshape(-1,1)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size = 0.2,
    random_state = 42
    )
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
result = []
hidden_sizes = [1, 2, 4, 8, 16]
def accuracy(model, X, y):
    predictions = model.predict(X) #prediciton function already executes a forward pass so we don't need prediction
    return np.mean(predictions == y)
def training_model(model, X, y, epochs):
    losses = []
    for epoch in range(epochs):
        y_pred = model.forward(X)
        loss = model.compute_loss(y_pred, y)
        losses.append(loss)
        model.backward(X, y)
        model.step()
    return losses
for hidden_size in hidden_sizes:
    model = TwoLayerMLP(
        input_size = 2,
        hidden_size = hidden_size,
        output_size = 1
        )
    results = {
        "hidden size": hidden_size,
        "loss" : training_model(model, X_train, y_train, 10000)[-1],  #only returns the last value of the losses list (remember losses has 1000(epochs) values, but we only want the last one
        "train accuracy" : accuracy(model, X_train, y_train),
        "test accuracy" : accuracy(model, X_test, y_test) #We only use the accuracy function on this because we never train on the testing data
        }  
    result.append(results)
print(result)
