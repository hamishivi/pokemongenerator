'''
A discriminator using the resnet architecture,
loaded with pretrained imagenet weights.
'''
import keras
from keras.utils import plot_model
from keras.models import Model
from keras.layers import Dense, Input, GlobalAveragePooling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.applications import nasnet

def make_discriminator(input_shape=None, demo=False):
    model_in = Input(shape=input_shape, name='input')
    nas = nasnet.NASNetMobile(input_tensor=model_in, include_top=False, pooling='avg')
    x = nas.output
    # just pass through dense for reshaping discriminator weights
    x = Dense(1024, activation='relu')(x)
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
    # load weights
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# Demo training
if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')) / 255
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = make_discriminator((28, 28, 1), demo=True)
    # no EM as that needs to be paired with the generator
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.Adam(lr=1e-4, epsilon=1e-8),
                  metrics=["accuracy"])

    model.fit(x_train, y_train,
              batch_size=128,
              epochs=32,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights('weights/resnet_disc_weights.hdf5')
