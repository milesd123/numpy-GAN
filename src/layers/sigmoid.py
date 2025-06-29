import numpy as np
from layer import Layer

class Sigmoid(Layer):
    def __init__(self):
        self.output = None

    def forward(self, input):
        # Apply sigmoid element wise across entire batch
        self.output = 1 / (1 + np.exp(-input))
        return self.output

    def backward(self, output_grad, learning_rate):
        # Sigmoid derivative element wise
        return output_grad * self.output * (1 - self.output)
