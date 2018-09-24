'''
A discriminator using the resnet architecture,
which won competitions in 2015.
This isn't state of the art, but I don't have the time nor
energy to write an SEnet or reneXt extension to this.
'''
import keras
from keras.models import Model
import keras.backend as K
from keras.layers import Dropout, Conv2D, Dense, BatchNormalization, Input, add, ZeroPadding2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D, MaxPooling2D

def identity_block(input_tensor, kernel_size, filters):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters3, (1, 1), padding='same')(x)
    x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = LeakyReLU()(x)
    return x


def conv_block(input_tensor, kernel_size, filters, strides=(2, 2)):
    filters1, filters2, filters3 = filters

    x = Conv2D(filters1, (1, 1), strides=strides)(input_tensor)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters2, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)

    x = Conv2D(filters3, (1, 1))(x)
    x = BatchNormalization()(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides)(input_tensor)
    shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = LeakyReLU()(x)
    return x


def make_discriminator(input_shape=None):
    model_in = Input(shape=input_shape)
    x = ZeroPadding2D(padding=(3, 3))(model_in)
    x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid')(model_in)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = ZeroPadding2D(padding=(1, 1))(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256])
    x = identity_block(x, 3, [64, 64, 256])

    x = conv_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])
    x = identity_block(x, 3, [128, 128, 512])

    x = conv_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])
    x = identity_block(x, 3, [256, 256, 1024])

    x = conv_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])
    x = identity_block(x, 3, [512, 512, 2048])

    x = Flatten()(x)

    x = Dense(1, activation='linear')(x)

    return Model(model_in, x)


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

# Example: mnist data set (digit recognition)
if __name__ == "__main__":
    from keras.datasets import mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')) / 255
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')) / 255
    y_test = keras.utils.to_categorical(y_test, num_classes=10)
    y_train = keras.utils.to_categorical(y_train, num_classes=10)
    model = make_discriminator((28, 28, 1))
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
