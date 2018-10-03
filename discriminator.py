'''
Baseline discriminator, a straightforward ConvNet.
'''
import keras
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.layers import Dropout, Conv2D, Dense, Flatten, BatchNormalization, Input, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

def make_discriminator(input_shape, demo=False):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(MaxPooling2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(LeakyReLU())

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D())
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(LeakyReLU())

    model.add(Conv2D(512, kernel_size=3, padding="same"))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=1))
    model.add(LeakyReLU())
    model.add(Dropout(0.25))

    model.add(Flatten())

    if demo:
        model.add(Dense(10, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear', name='output'))

    img = Input(shape=input_shape)
    validity = model(img)

    return Model(img, validity, name="discriminator")

# For pretrained models
def mnist(filepath):
    from keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    model = make_discriminator((28, 28, 1), demo=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.RMSprop(lr=0.00005),
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
              epochs=6,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
