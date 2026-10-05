import numpy as np
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
    
class TwoLayerMLP:

    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.1):
        self.learning_rate = learning_rate
        # Parameters
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))

        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
    def forward(self, X):

        # Layer 1
        self.Z1 = X @ self.W1 + self.b1
        self.A1 = np.tanh(self.Z1)
        # Layer 2
        self.Z2 = self.A1 @ self.W2 + self.b2
        self.A2 = sigmoid(self.Z2)
        return self.A2
    def compute_loss(self, y_hat, y):
        y_hat = np.clip(y_hat, 1e-8, 1 - 1e-8)
        return -np.mean(
            y * np.log(y_hat)
            + (1 - y) * np.log(1 - y_hat)
        )
    def backward(self, X, y):
        m = X.shape[0]
        dZ2 = self.A2 - y
        self.dW2 = (self.A1.T @ dZ2) / m
        self.db2 = np.sum(dZ2, axis=0, keepdims=True) / m
        dA1 = dZ2 @ self.W2.T
        dZ1 = dA1 * (1 - self.A1 ** 2)

        self.dW1 = (X.T @ dZ1) / m
        self.db1 = np.sum(dZ1, axis=0, keepdims=True) / m
    def step(self):
        self.W1 -= self.learning_rate * self.dW1
        self.b1 -= self.learning_rate * self.db1

        self.W2 -= self.learning_rate * self.dW2
        self.b2 -= self.learning_rate * self.db2

    def predict(self, X):
        probabilities = self.forward(X)
        return (probabilities >= 0.5).astype(int)
""" if __name__ == "__main__":
    np.random.seed(42)

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ], dtype=float)

    model = TwoLayerMLP(
        input_size=2,
        hidden_size=4,
        output_size=1,
        learning_rate=0.1
    )

    for epoch in range(10000):

        y_hat = model.forward(X)

        loss = model.compute_loss(y_hat, y)

        model.backward(X, y)

        model.step()

        if epoch % 1000 == 0:
            print(epoch, loss)

    print("Final probabilities:")
    print(model.forward(X))

    print("Final predictions:")
    print(model.predict(X)) """

