'''
Adapted from https://github.com/manicman1999/StyleGAN-Keras, with my own comments.

The mapping network is defined as part of the generator, rather than having its own file.
'''
from keras.layers import Conv2D, Dense, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Input, add, Cropping2D
from keras.models import Model

from AdaIN import AdaInstanceNormalization


def g_block(inp, style, noise, filter_size, upsample = True):
    # b and g = beta and gamma. these are learnt transformations for the 
    # AdaIN operation. Both comprise 'A' in the paper diagram.
    b = Dense(filter_size)(style)
    b = Reshape([1, 1, filter_size])(b)
    g = Dense(filter_size)(style)
    g = Reshape([1, 1, filter_size])(g)
    # add noise to block ('B' in the paper diagram)
    n = Conv2D(filters = filter_size, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    
    # upsample if we need to.
    # this part takes our input and puts it through the conv layer.
    if upsample:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    else:
        out = inp
    # add our output with the noise
    out = add([out, n])
    # AdaIN gets previous output + b and g (beta and gamma for transformation)
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(0.01)(out)
    # the second half of the styleGAN generator block
    # beta and gamma for our second AdaIN operations
    b = Dense(filter_size)(style)
    b = Reshape([1, 1, filter_size])(b)
    g = Dense(filter_size)(style)
    g = Reshape([1, 1, filter_size])(g)
    # add noise again ('B' in the paper diagram)
    n = Conv2D(filters = filter_size, kernel_size = 1, padding = 'same', kernel_initializer = 'he_normal')(noise)
    # pass output through conv
    out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal')(out)
    # apply AdaIN to output and thats it!
    out = AdaInstanceNormalization()([out, b, g])
    # add output and noise
    out = add([out, n])
    out = LeakyReLU(0.01)(out)
    return out


def make_generator(im_size, latent_size, num_channels):
    # we have multiple inputs, for each size
    # below is the mapping network 'rolled into'
    # the generator. It is much smaller than the mapping network
    # in the paper, mainly for faster training.
    style_input = Input(shape = [latent_size])
    style = Dense(512, kernel_initializer = 'he_normal')(style_input)
    style = LeakyReLU(0.1)(style)
    style = Dense(512, kernel_initializer = 'he_normal')(style)
    style = LeakyReLU(0.1)(style)
    # get the noise image and crop for each size
    inputs_noise = Input(shape = [im_size, im_size, 1])
    noises = [Activation('linear')(inputs_noise)]
    x = im_size
    while x > 4:
        x = int(x / 2)
        noises.append(Cropping2D(int(x/2))(noises[-1]))
    # our initial input
    inp = Input(shape = [1])
    x = Dense(4 * 4 * 512, kernel_initializer = 'he_normal')(inp)
    x = Reshape([4, 4, 512])(x)
    x = g_block(x, style, noises[-1], 512, upsample=False)
    # apply layers as need be
    if(im_size >= 1024):
        x = g_block(x, style, noises[7], 512)
    if(im_size >= 512):
        x = g_block(x, style, noises[6], 384)
    if(im_size >= 256):
        x = g_block(x, style, noises[5], 256)
    if(im_size >= 128):
        x = g_block(x, style, noises[4], 192)
    if(im_size >= 64):
        x = g_block(x, style, noises[3], 128)
    # final set of blocks
    x = g_block(x, style, noises[2], 64)
    x = g_block(x, style, noises[1], 32)
    x = g_block(x, style, noises[0], 16)
    # output
    x = Conv2D(filters = num_channels, kernel_size = 1, padding = 'same', activation = 'tanh')(x)
    # and then we have our model!
    return Model(inputs = [style_input, inputs_noise, inp], outputs = x)