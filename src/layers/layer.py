from abc import ABC, abstractmethod

class Layer(ABC):
    def forward(self, input):
        raise NotImplementedError

    def backward(self, output_grad, learning_rate):
        # Requires the output gradient from the downstream layer
        # Calculate the (gradient)partial derivative with respect to the weights of the current layer
        # Calculate the partial derivative with respect to the input (to the current layer)
        # to be used as for the upstream layer.
        # Change the weights.
        # Return the upstream gradient
        raise NotImplementedError