"""
A alternate generator setup, using transpose layers instead of
upsampling. This is used for all the non-pokemon generation.
(mnist, anime, cifar models)
"""
import keras
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, \
    BatchNormalization, Reshape, Input, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

# credit to https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
# for this generator design, although I have altered it a bit.
def make_mnist_generator(input_shape=(100,)):
    '''For the MNIST digit generation'''
    model = Sequential()

    model.add(Dense(256 * 7 * 7, input_dim=100, name='input'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(256, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(1, (5, 5), padding='same', activation='tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

# Design from https://github.com/tjwei/GANotebooks for the anime generator and 
# cifar generator. A similar discriminator was also used.
def make_anime_generator(input_shape=(40,)):
    '''For the anime face generation'''
    model = Sequential()

    model.add(Dense(1024 * 1 * 1, input_dim=40, name='input'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((1, 1, 1024)))

    model.add(Conv2DTranspose(1024, kernel_size=3, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size=4, padding='same', strides=2, activation='tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

def make_cifar_generator(input_shape=(100,)):
    '''For the cifar digit generation'''
    model = Sequential()

    model.add(Dense(1024 * 1 * 1, input_dim=100, name='input'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((1, 1, 1024)))

    model.add(Conv2DTranspose(1024, kernel_size=2, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size=4, padding='same', strides=2, activation='tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)
