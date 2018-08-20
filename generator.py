import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, Input
from keras.layers.advanced_activations import LeakyReLU

def make_generator():
    model = Sequential()

    # takes in 100dim noise vector as seed
    model.add(Dense(4 * 4 * 512, input_dim=100))

    model.add(Reshape((4, 4, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=(5, 5), strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=(5, 5), strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size=(5, 5), strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, kernel_size=(5, 5), strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    # Because we normalized training inputs to lie in the range [-1, 1],
    # the tanh function should be used for the output of the generator to ensure its output
    # also lies in this range.
    model.add(Conv2DTranspose(3, kernel_size=(5, 5), strides=[2,2], padding='same', data_format='channels_last'))
    model.add(Activation("tanh"))

    noise = Input(shape=(100,))
    img = model(noise)

    return Model(noise, img)

if __name__ == '__main__':
    print("TODO: make an image segmentation task?")
    make_generator()