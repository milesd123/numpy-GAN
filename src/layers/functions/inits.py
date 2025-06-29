import numpy as np

# Good for relu/prelu,
def kaiming_he_kernel(out_channels, in_channels, kernel_size):
    fan_in = in_channels * kernel_size
    std = np.sqrt(2.0 / fan_in)
    weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * std
    return weights

def kaiming_he_dense(input, dense_output):
    std = np.sqrt(2/input)
    weights = np.random.rand(dense_output, input) * std
    return weights