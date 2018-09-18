"""
A alternate generator setup, using transpose layers instead of
upsampling.
"""

import os
import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import UpSampling2D, Activation, Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, Input, MaxPooling2D

from keras.layers.advanced_activations import LeakyReLU

def make_generator(input_shape=(100,)):
    model = Sequential()

    # takes in 100dim noise vector as seed
    model.add(Dense(4 * 4 * 512, input_dim=100))
    model.add(Reshape((4, 4, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(16, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())


    model.add(Conv2DTranspose(3, kernel_size=5, strides=[2,2], padding='same'))
    model.add(Activation("tanh"))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

make_generator()