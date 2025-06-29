from generator import Generator
from ..layers.convolution import Convolution2D
from ..layers.prelu import PReLU
from ..layers.residual_blocks import ResidualBlocks
from ..layers.upsample_block import UpsampleBlock

training_data = None
training_labels = None

g = Generator()
g.add(Convolution2D(3, 64, 9, 4))
g.add(PReLU(num_channels=64))

#Residual Blocks
g.add(ResidualBlocks(channels=64, kernel_size=3, padding=1, n_blocks=16))

#Post residual connection
g.add(Convolution2D(in_channels=64, out_channels=64, padding=1, kernel_size=3))
g.add(UpsampleBlock(in_channels=64, upscale_factor=2, kernel_size=3, padding=1))
g.add(UpsampleBlock(in_channels=64, upscale_factor=2, kernel_size=3, padding=1))
g.add(Convolution2D(in_channels=64, out_channels=3, kernel_size=9, padding=4))

# See generator.py
g.fit(input_imgs=training_data, labels=training_labels, learning_rate=1e4, epochs=10, lambda_grad=.05)

g.save('generator_model')