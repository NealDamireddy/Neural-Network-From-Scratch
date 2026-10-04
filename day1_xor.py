import numpy as np
import matplotlib.pyplot as plt
learning_rate = 0.1

np.random.seed(42)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
], dtype=float)

Y = np.array([
    [0],
    [1],
    [1],
    [0]
], dtype=float)

m = X.shape[0]

input_size = 2
hidden_size = 4
output_size = 1
W1 = np.random.randn(input_size, hidden_size) * 0.5
b1 = np.zeros((1, hidden_size))

W2 = np.random.randn(hidden_size, output_size) * 0.5
b2 = np.zeros((1, output_size))
for epoch in range(10000):

    # 1. forward pass
    Z1 = X @ W1 + b1
    A1 = np.tanh(Z1)

    Z2 = A1 @ W2 + b2
    Y_hat = sigmoid(Z2)

    # 2. loss
    loss = -np.mean(
        Y * np.log(Y_hat + 1e-8)
        + (1 - Y) * np.log(1 - Y_hat + 1e-8)
    )

    # 3. backward pass
    dZ2 = Y_hat - Y
    dW2 = (A1.T @ dZ2) / m
    db2 = np.mean(dZ2, axis=0, keepdims=True)

    dA1 = dZ2 @ W2.T
    dZ1 = dA1 * (1 - A1**2)

    dW1 = (X.T @ dZ1) / m
    db1 = np.mean(dZ1, axis=0, keepdims=True)

    # 4. update parameters
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2
    if epoch % 1000 == 0:
        print(epoch, loss)
        predictions = (Y_hat >= 0.5).astype(int)

        print("Final predictions:")
        print(predictions)