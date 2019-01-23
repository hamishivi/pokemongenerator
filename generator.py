'''
Generator, using a mixture of upsampling and
convolutional layers instead of transpose layers.
Used for the pokemon model.
'''
import keras
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import UpSampling2D, Activation, Conv2D, Dense, \
    BatchNormalization, Reshape, Input, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

def make_generator(input_shape=(100,)):
    model = Sequential()
    # takes in 100dim noise vector as seed
    model.add(Dense(4 * 4 * 512, input_dim=100, name='input'))
    model.add(Reshape((4, 4, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(512, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(3, kernel_size=(5, 5), padding='same', data_format='channels_last'))
    model.add(Activation('tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)
