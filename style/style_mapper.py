'''
based on https://github.com/manicman1999/StyleGAN-Keras
The styleGAN introduces a third, unique network: the mapping network,
which is technically part of the generator but defined separately here
in this implementation.
'''
from keras.layers import Input, Dense, LeakyReLU
from keras.models import Model

# this is a really simple feedforward network :)
def make_mapper(latent_size):
    # only use 5 layers instead of 8 for faster training (TODO: test this!)
    inp = Input(shape = [latent_size])
    x = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(inp)
    x = LeakyReLU(0.01)(x)
    x = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x = LeakyReLU(0.01)(x)
    x = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)
    x = LeakyReLU(0.01)(x)
    x = Dense(512, kernel_initializer = 'he_normal', bias_initializer = 'zeros')(x)

    return Model(inputs = inp, outputs = x)
