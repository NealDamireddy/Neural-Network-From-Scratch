import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from mlp import TwoLayerMLP

np.random.seed(42)
model = TwoLayerMLP(
    input_size = 2, #creating a model that takes in 2 features per example, we need this for the size of the weight matrix
    hidden_size = 8, #This is how many hidden neurons we will have in the first layer
    output_size = 1,
    )
X, y = make_moons(
    n_samples = 500,
    noise = 0.15,
    random_state = 42
    )
for epoch in range(10000):
    y_hat = model.forward(X)
    loss = model.compute_loss(y_hat, y.reshape(-1, 1))
    model.backward(X, y.reshape(-1, 1))
    model.step() #optimize the weights over 10000 iterations -> this returns a matrix with the optimized weights, after you run this loop the weights are optimized for this training dataset. Any data after this will be optimized for the inital weights
y_hat = model.forward(X) #This is the prediction with the optimized weights
x1_values = np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 200) #defines the 
x2_values = np.linspace(X[:,0].min() - 0.5, X[:,0].max() + 0.5, 200)
xx, yy = np.meshgrid(x1_values, x2_values)
grid = np.c_[xx.ravel(), yy.ravel()]
model.forward(grid) # we are running the predicitons of every single value in the domain and range using the "optimized" weights
activation = model.A1[:,0] #this is the activation for neuron #1. We are looking at all 500 predictions so dimensions are (40000,1)
activation = activation.reshape(xx.shape)
print(activation.shape)
plt.contourf(
    xx,
    yy,
    activation,
    levels = 50)
plt.show()
