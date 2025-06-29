import numpy as np

def mse_loss_grad(pred, label):
    # Return the loss and the gradient.
    N, C, H, W = pred.shape
    return np.mean((pred-label)**2), ( 2/(N*C*H*W) ) * (pred - label)
