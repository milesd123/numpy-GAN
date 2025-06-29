from generator import Generator
from discriminator import Discriminator
import numpy as np
from ..layers.convolution import Convolution2D
from ..layers.flatten import Flatten
from ..layers.leakyReLU import leakyRELU
from ..layers.dense import Dense
from ..layers.batch_norm import BatchNorm
from ..layers.functions.total_loss_grad import loss_grad_generator
from ..layers.functions.bce import binary_cross_entropy
from ..layers.functions.vgg import vgg_extractor, vgg_loss
from ..layers.sigmoid import Sigmoid

#
dataset = None
labels = None

generator = Generator.load('generator_model.pkl')
discriminator = Discriminator()

# Add layers to discriminator
discriminator.add(Convolution2D(in_channels=3, out_channels=64, kernel_size=3, padding=1))
discriminator.add(leakyRELU(alpha=.2))
#block 1
discriminator.add(Convolution2D(in_channels=64, out_channels=64, kernel_size=3, padding=1))
discriminator.add(BatchNorm())
discriminator.add(leakyRELU(.2))
#block2
discriminator.add(Convolution2D(in_channels=64, out_channels=128, kernel_size=3, padding=1))
discriminator.add(BatchNorm())
discriminator.add(leakyRELU(.2))
#block3
discriminator.add(Convolution2D(in_channels=128, out_channels=128, kernel_size=3, padding=1))
discriminator.add(BatchNorm())
discriminator.add(leakyRELU(.2))
#block4
discriminator.add(Convolution2D(in_channels=128, out_channels=256, kernel_size=3, padding=1))
discriminator.add(BatchNorm())
discriminator.add(leakyRELU(.2))
#block5
discriminator.add(Convolution2D(in_channels=256, out_channels=256, kernel_size=3, padding=1))
discriminator.add(BatchNorm())
discriminator.add(leakyRELU(.2))
#block6
discriminator.add(Convolution2D(in_channels=256, out_channels=512, kernel_size=3, padding=1))
discriminator.add(BatchNorm())
discriminator.add(leakyRELU(.2))
X = None #TODO: Implement Stride, and stride down to 4x4 for flattening...
#flatten from 512*4*4 -> 8192
discriminator.add(Flatten)

discriminator.add(Dense(8192, 1024))
discriminator.add(leakyRELU(.2))
discriminator.add(Dense(1024, 1))
discriminator.add(Sigmoid())

#Example training data
epochs = 100
batch_size = 4
learning_rate = 1e-4
lambda_grad = 0.05

for epoch in range(epochs):
    for batch_idx in range(0, len(dataset), batch_size):
        # Get Data
        low_res = dataset[batch_idx:batch_idx+batch_size]     # (N, 3, 64, 64)
        high_res = labels[batch_idx:batch_idx+batch_size]     # (N, 3, 256, 256)

        # Train Discriminator
        gen_imgs = generator.forward(low_res)

        real_preds = discriminator.forward(high_res)
        fake_preds = discriminator.forward(gen_imgs)

        real_labels = np.ones_like(real_preds)
        fake_labels = np.zeros_like(fake_preds)

        real_loss, real_grad = binary_cross_entropy(real_preds, real_labels)
        fake_loss, fake_grad = binary_cross_entropy(fake_preds, fake_labels)

        disc_loss = real_loss + fake_loss
        disc_grad = real_grad + fake_grad

        discriminator.backward(disc_grad, learning_rate)

        print(f"[D] Epoch {epoch+1} Batch {batch_idx//batch_size+1} | Loss: {disc_loss:.4f}")

        # Train Generator
        gen_imgs = generator.forward(low_res)
        fake_preds = discriminator.forward(gen_imgs)
        valid_labels = np.ones_like(fake_preds)

        # Adversarial + perceptual loss
        adv_loss, adv_grad = binary_cross_entropy(fake_preds, valid_labels)
        perceptual_loss, perceptual_grad = vgg_loss(gen_imgs, high_res, vgg_extractor)

        total_loss = perceptual_loss + lambda_grad * adv_loss
        total_grad = perceptual_grad + lambda_grad * adv_grad

        generator.backward(total_grad, learning_rate)

        print(f"[G] Epoch {epoch+1} Batch {batch_idx//batch_size+1} | Loss: {total_loss:.4f}")

generator.save('GAN_trained_generator')
discriminator.save('GAN_trained_discriminator')