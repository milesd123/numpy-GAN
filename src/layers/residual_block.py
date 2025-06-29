from layer import Layer
from src.layers.convolution import Convolution2D
from src.layers.prelu import PReLU

class ResidualBlock(Layer):
    def __init__(self, channels, kernel_size, padding):
        """
        Residual block: convolute, prelu, convolute, residual connection (adding input to output)

        :param channels:
        :param kernel_size:
        :param padding:
        """
        self.conv1 = Convolution2D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.conv2 = Convolution2D(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding)
        self.prelu = PReLU(num_channels=channels)

    def forward(self, input):
        output = self.conv1.forward(input)
        output = self.prelu.forward(output)
        output = self.conv2.forward(output)

        #Residual connection, add the input to the output.
        return output + input

    def backward(self, output_grad, learning_rate):
        #Reverse

        grad_conv2 = self.conv2.backward(output_grad, learning_rate)
        grad_prelu = self.prelu.backward(grad_conv2, learning_rate)
        grad_conv1 = self.conv1.backward(grad_prelu, learning_rate)

        #Residual connection
        return grad_conv1 + output_grad
