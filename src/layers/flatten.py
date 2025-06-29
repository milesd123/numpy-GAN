from layer import Layer

class Flatten(Layer):
    def __init__(self):
        self.input_shape = None
    def forward(self, input):
        #If we have a tensor (Batch, Channel, H, W), we want to flatten
        # to (Batch, C * H * W)
        self.input_shape = input.shape
        N = input.shape[0]

        return input.reshape(N, -1)

    def backward(self, output_grad, learning_rate):
        return output_grad.reshape(self.input_shape)
