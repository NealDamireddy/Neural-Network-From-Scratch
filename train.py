import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

from mlp import TwoLayerMLP

np.random.seed(42)
X, y = make_moons(
    n_samples = 500,
    noise = 0.15,
    random_state = 42
    )
y_axis = []
x_axis = []
y = y.reshape(-1,1)
model = TwoLayerMLP(
    input_size = 2, #creating a model that takes in 2 features per example, we need this for the size of the weight matrix
    hidden_size = 8, #This is how many hidden neurons we will have in the first layer
    output_size = 1,
    )
for epoch in range(1000):
    y_hat = model.forward(X)
    loss = model.compute_loss(y_hat, y)
    model.backward(X, y)
    model.step()
    y_axis.append(loss)
    x_axis.append(epoch)
    if epoch % 100 == 0:
        print(epoch, loss)
prediction_accuracy = np.mean(model.predict(X) == y)
x1_min = np.min(X[:,0]) - 0.5
x1_max = np.max(X[:,0]) + 0.5
x2_min = np.min(X[:,1]) - 0.5
x2_max = np.max(X[:,1]) + 0.4
x1_values = np.linspace(x1_min, x1_max, 200) #creating array of 200 points with minimum at xmin and max at xmax
x2_values = np.linspace(x2_min, x2_max, 200) #same thing here we are creating another array, this will form the 2D plane we can use to visualize the decision boundary
xx, yy = np.meshgrid(x1_values, x2_values) #creating a grid of points that corresponds each array index with each other
final_plane = np.c_[xx.ravel(),yy.ravel()] #taking the values of the xx 2D arrays and yy 2D arrays and flattening them, then turning them into columns and adding them together
prediction_plane = model.predict(final_plane)
prediction_plane = prediction_plane.reshape(xx.shape) #you want to reshape the outcome of the predictions on the first run to the shape of the axes
plt.contourf(xx, yy, prediction_plane)
plt.show()

        
