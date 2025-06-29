import pickle
import numpy as np
import time
from ..layers.functions.total_loss_grad import loss_grad_generator

class Generator:
    def __init__(self):
        self.layers = [] # Holds our layer objects

    def add(self, layer):
        # Add a layer to the layer list.
        self.layers.append(layer)

    def forward(self, input_data):
        # Solo forward method for generator + discriminator training
        output = input_data

        # Loop through the layer's forward method
        for layer in self.layers:
            output = layer.forward(output)

        #Clip the output between 0,1 most values will be in the range anyways
        return np.clip(output, 0, 1)

    def backward(self, grad, learning_rate):
        # Solo backward method for generator + discriminator training
        for layer in reversed(self.layers):
            grad = layer.backward(grad, learning_rate)

    def fit(self, input_imgs, labels, learning_rate=0.001, epochs=1, batch_size=1, lambda_grad=1.0):
        # Get sample and batch amount
        num_samples = len(input_imgs)
        num_batches = int(np.ceil(num_samples / batch_size))
        start = time.time()

        # Start training through epochs
        for epoch in range(epochs):
            for batch_idx in range(num_batches):
                # Get batch indexes
                batch_start = batch_idx * batch_size
                batch_end = min((batch_idx + 1) * batch_size, num_samples)

                input_batch = input_imgs[batch_start:batch_end]
                label_batch = labels[batch_start:batch_end]

                # Forward
                output = self.forward(input_batch)

                # Compute Loss and Gradient, loss_grad_generator uses VGG loss and MSE loss
                loss, grad = loss_grad_generator(output, label_batch, lambda_grad)

                # backpropogration
                self.backward(grad, learning_rate)

                print(f"[G] Epoch {epoch+1}/{epochs} | Batch {batch_idx+1}/{num_batches} | "
                      f"Loss: {loss:.4f} | Time: {(time.time() - start)/60:.2f} min")

    #Pickle for model saving and loading
    def save(self, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)

