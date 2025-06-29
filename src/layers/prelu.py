import numpy as np
from layer import Layer

class PReLU(Layer):
    def __init__(self, num_channels):
        self.input = None
        self.alpha = np.full((num_channels, 1, 1), 0.25)  # one 'a' per channel
        self.grad_a = 0.0        # Accumulates gradient wrt a

    def forward(self, input):
        self.input = input
        # np.where(condition, x, y)
        return np.where(input >= 0, input, self.alpha * input)

    def backward(self, output_grad, learning_rate):
        # Derivative wrt input
        grad_input = np.where(self.input >= 0, 1, self.alpha) * output_grad

        # Derivative wrt a: only from negative input positions
        self.grad_alpha = np.sum(output_grad * self.input * (self.input < 0), axis=(0, 2, 3), keepdims=True)

        # Update weights
        self.alpha -= learning_rate * self.grad_alpha

        return grad_input
