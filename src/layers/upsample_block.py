from layer import Layer
from pixel_shuffle import PixelShuffle
from convolution import Convolution2D
from prelu import PReLU

class UpsampleBlock(Layer):
    def __init__(self, in_channels: int,upscale_factor: int, kernel_size: int, padding: int):
        self.conv = Convolution2D(in_channels = in_channels,
                                  out_channels = in_channels * (upscale_factor**2),
                                  kernel_size = kernel_size,
                                  padding = padding)
        self.prelu = PReLU()
        self.pixelshuffle = PixelShuffle(upscale_factor)

    def forward(self, input):
        input = self.conv.forward(input) # Create more geature maps
        # Distribute the square of the upscale factor into the output maps
        # Reducing the feature maps, expanding the H,W
        input = self.pixelshuffle.forward(input)
        return self.prelu.forward(input)

    def backward(self, output_grad, learning_rate):
        output_grad = self.prelu.backward(output_grad, learning_rate)
        output_grad = self.pixelshuffle.backward(output_grad, learning_rate)
        return self.conv.backward(output_grad, learning_rate)