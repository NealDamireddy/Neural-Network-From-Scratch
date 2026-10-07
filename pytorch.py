import torch

N = 10
dim_input= 1
dim_output = 1

X = torch.randn(N, dim_input) #create tensor object with 10 examples and 1 input per example

W_true = torch.tensor([2.0])
b_true = torch.tensor([1.0])
y_true =(X @ W_true + b_true) * 0.1 #We want to create the true value of y by using the input X with the optimized weights

W1 = torch.randn(dim_input, dim_output, requires_grad = True) #creating weight and bias tensors that must be adjusted to the optimal by comparing it to y_true
b1 = torch.zeros(1, requires_grad = True) #make sure you include requires_grad = true for tensors that are changing

y_hat = X @ W1 + b
y_hat = y_hat.forward()
learning_rate = 0.001
for epoch in range(5000):
    y_prediction = X @ W + b1
    loss = torch.mean((y_hat - y_true)**2)

    loss.backwards()
    with torch.no_grad:
        W1 -= learning_rate * W1.grad, b1 -= learning_rate * b1.grad #nudge parameters in correct direction -> gradient descent
    W1.grad_zero_(), b1.grad.zero_() #reset for next round of learnings

    
