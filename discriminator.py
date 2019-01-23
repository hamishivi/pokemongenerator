'''
Discriminators used across all WGAN setups.
'''
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Conv2D, Dense, Flatten, BatchNormalization, Input, add, Lambda, concatenate, MaxPool2D
from keras_layer_normalization import LayerNormalization
from keras.layers.advanced_activations import LeakyReLU

def make_discriminator(input_shape, batchnorm=True):
    model = Sequential()

    model.add(Conv2D(128, kernel_size=4, strides=2, input_shape=input_shape, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(256, kernel_size=4, strides=2, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(512, kernel_size=4, strides=2, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(1024, kernel_size=4, strides=2, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Flatten())

    model.add(Dense(1, activation='linear', name='output'))

    img = Input(shape=input_shape)
    validity = model(img)

    return Model(img, validity, name="discriminator")

def make_poke_discriminator(input_shape, batchnorm=True):
    model = Sequential()

    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=input_shape, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(128, kernel_size=5, strides=2, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(256, kernel_size=5, strides=2, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(512, kernel_size=5, strides=2, padding="same"))
    if batchnorm:
        model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Flatten())

    model.add(Dense(1, activation='linear', name='output'))

    img = Input(shape=input_shape)
    validity = model(img)

    return Model(img, validity, name="discriminator")