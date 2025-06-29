from layer import Layer
from residual_block import ResidualBlock

class ResidualBlocks(Layer):
    def __init__(self, channels, kernel_size, padding, n_blocks):
        #So that its easier in training, we can use this rather#
        # than manually adding 16 blocks or creating a loop there.

        self.blocks = [
            ResidualBlock(channels, kernel_size, padding)
            for _ in range(n_blocks)
        ]

    def forward(self, input):
        out = input
        for block in self.blocks:
            out = block.forward(out)
        return out

    def backward(self, output_grad, learning_rate):
        grad = output_grad
        for block in reversed(self.blocks):
            grad = block.backward(grad, learning_rate)
        return grad
