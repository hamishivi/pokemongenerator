from keras.layers import Conv2D, Dense, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Input, add, Cropping2D
from keras.models import Model

from AdaIN import AdaInstanceNormalization

# generator block, defined separately as it is very complicated!
def g_block(inp, style, noise, filter_size, upsample = True):
    # b and g = beta and gamma. these are learnt transformations for the 
    # AdaIN operation. Both comprise 'A' in the paper diagram.
    b = Dense(filter_size, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
    b = Reshape([1, 1, filter_size])(b)
    g = Dense(filter_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
    g = Reshape([1, 1, filter_size])(g)
    # add noise to block ('B' in the paper diagram)
    n = Conv2D(filters = filter_size, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)
    
    # upsample if we need to.
    # this part takes our input and puts it through the conv layer.
    if upsample:
        out = UpSampling2D(interpolation = 'bilinear')(inp)
        out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)
    else:
        out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    # add our output with the noise
    out = add([out, n])
    # AdaIN gets previous output + b and g (beta and gamma for transformation)
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(0.01)(out)
    # the second half of the styleGAN generator block
    # beta and gamma for our second AdaIN operations
    b = Dense(filter_size, kernel_initializer = 'he_normal', bias_initializer = 'ones')(style)
    b = Reshape([1, 1, filter_size])(b)
    g = Dense(filter_size, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(style)
    g = Reshape([1, 1, filter_size])(g)
    # add noise again ('B' in the paper diagram)
    n = Conv2D(filters = filter_size, kernel_size = 1, padding = 'same', kernel_initializer = 'zeros', bias_initializer = 'zeros')(noise)
    # pass output through conv
    out = Conv2D(filters = filter_size, kernel_size = 3, padding = 'same', kernel_initializer = 'he_normal', bias_initializer = 'zeros')(out)
    # add output and noise
    out = add([out, n])
    # apply AdaIN to output and thats it!
    out = AdaInstanceNormalization()([out, b, g])
    out = LeakyReLU(0.01)(out)
    return out


def make_generator(im_size, latent_size):
    # we have multiple inputs, for each size
    # below is the mapping network 'rolled into'
    # the generator
    style_input = Input(shape = [latent_size])
    style = Dense(512, kernel_initializer = 'he_normal')(style_input)
    style = LeakyReLU(0.01)(style)
    style = Dense(512, kernel_initializer = 'he_normal')(style)
    style = LeakyReLU(0.01)(style)
    # style = Dense(512, kernel_initializer = 'he_normal')(style)
    # style = LeakyReLU(0.01)(style)
    # style = Dense(512, kernel_initializer = 'he_normal')(style)
    # style = LeakyReLU(0.01)(style)
    # style = Dense(512, kernel_initializer = 'he_normal')(style)
    # style = LeakyReLU(0.01)(style)
    # style = Dense(512, kernel_initializer = 'he_normal')(style)
    # style = LeakyReLU(0.01)(style)
    # style = Dense(512, kernel_initializer = 'he_normal')(style)
    # get the noise image and crop for each size
    inputs_noise = Input(shape = [im_size, im_size, 1])
    noises = [Activation('linear')(inputs_noise)]
    x = im_size
    while x > 4:
        x = int(x / 2)
        noises.append(Cropping2D(int(x/2))(noises[-1]))
    # our initial input
    inp = Input(shape = [1])
    x = Dense(4 * 4 * im_size, kernel_initializer = 'he_normal')(inp)
    x = Reshape([4, 4, im_size])(x)
    x = g_block(x, style, noises[-1], im_size, upsample=False)
    # apply layers as need be (TODO: make this less ugly)
    if(im_size >= 1024):
        x = g_block(x, style, noises[7], 512) # Size / 64
    if(im_size >= 512):
        x = g_block(x, style, noises[6], 384) # Size / 64
    if(im_size >= 256):
        x = g_block(x, style, noises[5], 256) # Size / 32
    if(im_size >= 128):
        x = g_block(x, style, noises[4], 192) # Size / 16
    if(im_size >= 64):
        x = g_block(x, style, noises[3], 128) # Size / 8
    # final set of blocks
    x = g_block(x, style, noises[2], 64) # Size / 4
    x = g_block(x, style, noises[1], 32) # Size / 2
    x = g_block(x, style, noises[0], 16) # Size
    # output
    x = Conv2D(filters = 3, kernel_size = 1, padding = 'same', activation = 'sigmoid')(x)
    # and then we have our model!
    return Model(inputs = [style_input, inputs_noise, inp], outputs = x)