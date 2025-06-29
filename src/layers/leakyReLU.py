import numpy as np
from layer import Layer

class leakyRELU(Layer):
    def __init__(self, alpha):
        self.input = None
        self.alpha = alpha

    def forward(self, input):
        self.input = input
        # np.where(condition, x, y)
        return np.where(input > 0, input, self.alpha * input)

    def backward(self, output_grad, learning_rate):
        dx = np.where(self.input > 0, 1, self.alpha)
        return output_grad * dx