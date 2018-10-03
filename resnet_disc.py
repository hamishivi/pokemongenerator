'''
A discriminator using the resnet architecture,
which won competitions in 2015. This is based off the
code found at https://github.com/keras-team/keras-applications/blob/master/keras_applications/resnet50.py,
which is the official keras implementation of resnet.
Due to memory and time constraints, I could not fit the entire model into
my GPU.
'''
import keras
from keras.utils import plot_model
from keras.models import Model
import keras.backend as K
from keras.layers import Dropout, Conv2D, Dense, BatchNormalization, Input, add, ZeroPadding2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D

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
    x = Conv2D(fil, (1, 1), strides=(2,2))(input_model)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(fil, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(fil*4, (1, 1))(x)
    x = BatchNormalization()(x)

    # shortcut route - required to downsample the image
    y = Conv2D(fil*4, (1, 1), strides=(2,2))(input_model)
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

# we feed +1 as label for real and -1 for fake images
# in the D, and opposite in the G.
def EM_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def compile_wasserstein_critic(model):
    model.compile(loss=EM_loss,
        optimizer=keras.optimizers.RMSprop(lr=0.00005),
        metrics=["accuracy"])

def compile_demo(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=1e-4, epsilon=1e-8),
        metrics=["accuracy"])

def resnet_mnist(filepath):
    from keras.datasets import mnist

    (_, _), (x_test, y_test) = mnist.load_data()
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    model = make_discriminator((28, 28, 1), demo=True)
    model.load_weights(filepath, by_name=False)
    # load weights
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


# Example: mnist data set (digit recognition)
if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')) / 255
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = make_discriminator((28, 28, 1), demo=True)
    # no EM as that needs to be paired with the generator
    compile_demo(model)

    tbCallBack = keras.callbacks.TensorBoard(log_dir='./log', histogram_freq=0, write_graph=True, write_images=True)


    model.fit(x_train, y_train,
          batch_size=128,
          epochs=6,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[tbCallBack])
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    model.save_weights('weights/resnet_disc_weights.hdf5')
