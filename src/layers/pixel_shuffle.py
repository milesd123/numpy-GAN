import numpy as np
from layer import Layer

class PixelShuffle(Layer):
    def __init__(self, upscale_factor):
        self.r = upscale_factor
        self.input_shape = None

    def forward(self, input):
        # Input shape: (C*r^2, H, W)
        self.input_shape = input.shape
        N, C_r2, H, W = input.shape
        r = self.r

        assert C_r2 % (r * r) == 0, "Channel count must be divisible by upscale_factor^2"
        C = C_r2 // (r * r)

        # reshape to (C, r, r, H, W)
        x = input.reshape(N, C, r, r, H, W)

        # transpose to (C, H, r, W, r)
        x = x.transpose(0, 1, 4, 2, 5, 3)

        # reshape to (C, H*r, W*r)
        out = x.reshape(N, C, H * r, W * r)
        return out

    def backward(self, grad_output, learning_rate):
        # grad_output shape: (N, C, H*r, W*r)
        N, C, Hr, Wr = grad_output.shape
        r = self.r
        H = Hr // r
        W = Wr // r
        Cr2 = C * r * r

        #  reshape to (N, C, H, r, W, r)
        x = grad_output.reshape(N, C, H, r, W, r)

        # transpose to (N, C, r, r, H, W)
        x = x.transpose(0, 1, 3, 5, 2, 4)

        # reshape to (N, C*r^2, H, W)
        grad_input = x.reshape(N, Cr2, H, W)
        return grad_input

