from layer import Layer
import numpy as np
from functions.inits import kaiming_he_kernel
from scipy.signal import correlate2d
from scipy.signal import convolve2d

class Convolution2D(Layer):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size  # int
        self.kernel_amount = out_channels
        self.padding = padding

        self.input = None
        self.kernel_matrix = kaiming_he_kernel(self.out_channels, self.in_channels, self.kernel_size)  # shape: (C_out,C_in, kH, kW)
        self.bias_matrix = kaiming_he_kernel((1, 1), self.kernel_amount).squeeze()     # shape: (C_out,)

    def forward(self, input):
        # input shape: (C_in, H, W)
        self.input = input

        N, C_in, H_in, W_in = input.shape
        C_out, _, kH, kW = self.kernel_matrix.shape

        # Padding
        if self.padding > 0:
            # Pad each channel separately with zeros
            padded_input = np.pad(
                input,
                ((0, 0), (0,0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant',
                constant_values=0,
            )
        else:
            padded_input = input

        H_out = H_in + 2 * self.padding - kH + 1
        W_out = W_in + 2 * self.padding - kW + 1

        output = np.zeros((N, C_out, H_out, W_out))
        for n in range(N):
            for out_c in range(C_out): # Loop throgh every output channel
                acc = np.zeros((H_out, W_out)) # initialize each new feature map
                for in_c in range(C_in): # Loop through each input channel
                    # Correlate each output channel with every input channel.

                    # Get the kernel for the current output channel at every input channel
                    kernel = self.kernel_matrix[out_c, in_c]
                    # correlate that kernel with each of the input feature maps/channels
                    acc += correlate2d(padded_input[n, in_c], kernel, mode='valid')

                output[n, out_c] = acc + self.bias_matrix[out_c] # Add the bias for the current output channel

        return output

    def backward(self, output_grad, learning_rate):
        """
        kernel gradient: correlate the input with the output gradient. dL/dK = input correlate dL/dY
        bias gradient: sum of each output channel
        upstream/input gradient: full correlation of the 180 rotated kernel matrix with the output gradient,
        AKA a full convolution. dL/dX = dL/dY *full K
        """

        #Initialize the gradient matrices with same size as the
        kernel_gradient = np.zeros_like(self.kernel_matrix)
        bias_gradient = np.zeros_like(self.bias_matrix)

        #Shape for loops
        N, C_in, _, _ = self.input.shape
        C_out, _, _, _ = self.kernel_matrix.shape
        if self.padding > 0:
            padded_input = np.pad(
                self.input,
                ((0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)),
                mode='constant'
            )
        else:
            padded_input = self.input

        upstream_grad = np.zeros_like(output_grad)
        for n in range(N):
            for c_out in range(C_out):
                for c_in in range(C_in):
                    # Gradient with respect to the kernel/weights
                    kernel_gradient[c_out, c_in] += correlate2d(padded_input[n, c_in], output_grad[n, c_out], mode='valid')

                    #Gradient with respect to the input. Notice how it is a full convolution
                    kernel_rotated = np.rot90(self.kernel_matrix[c_out, c_in], 2)
                    upstream_grad[n, c_in] += convolve2d(output_grad[n, c_out], kernel_rotated, mode='full')

                #Gradient with respect to the bias
                bias_gradient[c_out] = np.sum(output_grad[n, c_out])

        #Update parameters
        self.kernel_matrix = self.kernel_matrix - learning_rate * kernel_gradient
        self.bias_matrix = self.bias_matrix - learning_rate * bias_gradient

        return upstream_grad