import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

X, y = make_moons(
    n_samples = 500,
    noise = 0.15,
    random_state = 42
    )
y = y.reshape(-1,1)

from mlp import TwoLayerMLP
np.random.seed(42)
model = TwoLayerMLP(
    input_size = 2, #creating a model that takes in 2 features per example, we need this for the size of the weight matrix
    hidden_size = 8, #This is how many hidden neurons we will have in the first layer
    output_size = 1,
    )
prediction = model.forward(X)
print(X.shape, y.shape, prediction.shape)
epsilon = 1e-5
def gradient_check(row,col, X, y):
    weight = model.W1[row, col]
    y_hat = model.forward(X)
    model.backward(X, y)
    analytic_gradient  = model.dW1[row, col] 
    W_one = weight + epsilon
    W_two = weight - epsilon
    model.W1[row,col] = W_one #set the first weight on W1 to our changed value
    forward_plus = model.forward(X) #This is our forward prediction (y_hat)
    loss_plus = model.compute_loss(forward_plus, y)
    model.W1[row, col ] = W_two
    forward_minus = model.forward(X)
    loss_minus = model.compute_loss(forward_minus, y)
    model.W1[row,col] = weight
    return (loss_plus - loss_minus)/(2* epsilon), analytic_gradient
numeric, analytic = gradient_check(0, 0, X, y)
print("Numerical:", numeric)
print("Backprop:", analytic)
numeric, analytic = gradient_check(0, 4, X, y)
print("Numerical:", numeric)
print("Backprop:", analytic)
numeric, analytic = gradient_check(1, 2, X, y)
print("Numerical:", numeric)
print("Backprop:", analytic)
numeric, analytic = gradient_check(1, 7, X, y)
print("Numerical:", numeric)
print("Backprop:", analytic)
    
