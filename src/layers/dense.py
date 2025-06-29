from layer import Layer
import numpy as np
from functions.inits import kaiming_he_dense

class Dense(Layer):
    def __init__(self, input_size, n_output_neurons):
        #Input size = number of channels after global pooling
        self.output_neurons = n_output_neurons
        self.input_size = input_size

        self.input = None
        self.weights = kaiming_he_dense(self.input_size, self.output_neurons)
        self.biases = np.zeros(self.output_neurons)

    def forward(self, input):
        # We will use these after average pooling, so the shape will only be 1 dimension
        self.input = input #shape (X,)
        output = input @ self.weights + self.biases
        return output

    def backward(self, output_grad, learning_rate):
        """
        For the dense layer, we can find the derivative of the loss with respect to the weights
        as dL/dW = dL/dY • input. For the biases, dL/dB = dL/dY, and the upstream gradient
        for the next backprop layer dL/dX = dL/dY • Weights
        """

        #compute the gradients
        #(in, N) @ (N, out) -> (in, out)
        weight_gradient = self.input.T * output_grad

        bias_gradient = np.sum(output_grad, axis=0)

        #(N, out) @ (out, in) -> (N, in)
        upstream_grad = output_grad @ self.weights.T

        #update weights
        self.weights -= learning_rate * weight_gradient
        self.biases -= learning_rate  * bias_gradient

        return upstream_grad
