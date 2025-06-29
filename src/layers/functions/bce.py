import numpy as np

def binary_cross_entropy(pred, label):

    #Avoid log(0)
    pred = np.clip(pred, 1e-12, 1 - 1e-12)

    #BCE Loss
    loss = -(label * np.log(pred) + (1 - label) * np.log(1-pred))
    loss = np.mean(loss)

    #BCE Gradient
    grad = (-(label / pred) + (1 - label) / (1 - pred)) / label.size
    return loss, grad
