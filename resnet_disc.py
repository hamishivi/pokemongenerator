'''
A discriminator using the resnet architecture,
which won competitions in 2015. This is largely based on the
code found at https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py,
which is the official keras implementation of resnet.
Due to memory and time constraints, I could not fit the entire model into
my GPU. This reduced model is the resnet discriminator model used in the
report.
'''
import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Conv2D, Dense, BatchNormalization, Input, add, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import MaxPooling2D

def identity_block(input_model, fil):
    x = Conv2D(fil, (1, 1), padding='same')(input_model)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(fil, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(fil*4, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, input_model])
    return LeakyReLU()(x)


def conv_block(input_model, fil):
    # route 1
    x = Conv2D(fil, (1, 1), strides=(2, 2))(input_model)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(fil, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(fil*4, (1, 1))(x)
    x = BatchNormalization()(x)

    # shortcut route - required to downsample the image
    y = Conv2D(fil*4, (1, 1), strides=(2, 2))(input_model)
    y = BatchNormalization()(y)

    # the paths converge
    x = add([x, y])
    x = LeakyReLU()(x)
    return x


def make_discriminator(input_shape=None, demo=False):
    model_in = Input(shape=input_shape, name='input')
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(model_in)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = MaxPooling2D((3, 3), strides=(2, 2), padding='valid')(x)

    x = conv_block(x, 64)
    x = identity_block(x, 64)

    x = conv_block(x, 128)
    x = identity_block(x, 128)

    x = conv_block(x, 256)
    x = identity_block(x, 256)

    x = conv_block(x, 512)
    x = identity_block(x, 512)
    x = Flatten()(x)

    if demo:
        x = Dense(10, activation='softmax', name='output')(x)
    else:
        x = Dense(1, activation='linear', name='output')(x)

    return Model(inputs=model_in, outputs=x)

# for loading a pretrained model
def resnet_mnist(filepath):
    from keras.datasets import mnist
    # setup stuff
    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    model = make_discriminator((28, 28, 1), demo=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=1e-4, epsilon=1e-8),
                  metrics=["accuracy"])
    model.load_weights(filepath, by_name=False)

    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')) / 255
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = make_discriminator((28, 28, 1), demo=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.00005),
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=12,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights('weights/improved_mnist.h5')
