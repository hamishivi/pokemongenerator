'''
Adapted from https://github.com/manicman1999/StyleGAN-Keras more or less completely.

The StyleGAN paper specifically mentions that its main contributions are in the generator,
so theoretically this discriminator could be swapped with different architectures if you wish.
'''
from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU
from keras.layers import Dropout, Flatten, Input
from keras.models import Model

def d_block(inp, filter_size, pooling = True):
    out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(inp)
    out = LeakyReLU(0.01)(out)
    if pooling:
        out = AveragePooling2D()(out)
    out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    out = LeakyReLU(0.01)(out)
    
    return out

def make_discriminator(im_size, num_channels):
    # basic input 
    inp = Input(shape = [im_size, im_size, num_channels])
    # stacking the discriminator blocks
    x = d_block(inp, 16)
    x = d_block(x, 32)
    x = d_block(x, 64
    if (im_size > 32):
        x = d_block(x, 128)
    if (im_size > 64):
        x = d_block(x, 192)
    if (im_size > 128):
        x = d_block(x, 256)
    if (im_size > 256):
        x = d_block(x, 384)
    if (im_size > 512):
        x = d_block(x, 512)
    # flatten it all down for the scoring
    x = Flatten()(x)
    # final dense blocks
    x = Dense(128)(x)
    x = LeakyReLU(0.01)(x)
    
    x = Dense(1)(x)
    
    return Model(inputs = inp, outputs = x)
