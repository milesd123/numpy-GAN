import numpy as np
from layer import Layer

class BatchNorm(Layer):
    def __init__(self):
        #Trainable parameters used to scale and shift our output. Y = gamma*X + beta
        self.gamma = None # Shift
        self.beta = None # Scale
        self.input = None

        #share variable for back prop
        self.mean = None
        self.variance = None
        self.normalized = None

    def forward(self, input):
        # input shape: (N, C, H, W)
        N, C, H, W = input.shape
        self.input = input

        if self.gamma is None:
            self.gamma = np.ones((C, 1, 1))
            self.beta = np.zeros((C, 1, 1))

        # Compute mean andvariance per channel over N, H, W
        self.mean = np.mean(input, axis=(0, 2, 3), keepdims=True)  # shape: (1, C, 1, 1)
        self.variance = np.var(input, axis=(0, 2, 3), keepdims=True)  # shape: (1, C, 1, 1)

        #normalize
        self.normalized = (input - self.mean) / np.sqrt(self.variance + 1e-5)
        output = self.gamma * self.normalized + self.beta
        return output

    def backward(self, output_grad, learning_rate):
        # Added batche support
        N, C, H, W = output_grad.shape
        M = N * H * W

        #https://arxiv.org/pdf/1502.03167v3 for the derivation
        std_inv = 1.0 / np.sqrt(self.variance + 1e-5)

        dxhat = output_grad * self.gamma
        x_mu = self.input - self.mean

        dvar = np.sum(dxhat * x_mu * -0.5 * std_inv ** 3, axis=(0, 2, 3), keepdims=True)
        dmu = np.sum(dxhat * -std_inv, axis=(0, 2, 3), keepdims=True) + dvar * np.mean(-2.0 * x_mu, axis=(0, 2, 3),
                                                                                       keepdims=True)

        upstream_grad = dxhat * std_inv + dvar * 2 * x_mu / M + dmu / M

        #parameter gradients
        gamma_grad = np.sum(output_grad * self.normalized, axis=(0, 2, 3), keepdims=True)
        beta_grad = np.sum(output_grad, axis=(0, 2, 3), keepdims=True)

        self.gamma -= learning_rate * gamma_grad
        self.beta -= learning_rate * beta_grad

        return upstream_grad

