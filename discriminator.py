'''
Baseline discriminator, a straightforward ConvNet.
'''
import keras
from keras.models import Sequential, Model
from keras.layers import Dropout, Conv2D, Dense, Flatten, BatchNormalization, Input, add, Lambda, concatenate, MaxPool2D
from keras_layer_normalization import LayerNormalization
from keras.layers.advanced_activations import LeakyReLU

def make_discriminator(input_shape, demo=False, batchnorm=True):
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

    if demo:
        model.add(Dense(10, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear', name='output'))

    img = Input(shape=input_shape)
    validity = model(img)

    return Model(img, validity, name="discriminator")

cardinality = 32

def add_common_layers(y):
    y = LayerNormalization()(y)
    y = LeakyReLU()(y)
    return y

def grouped_convolution(y, nb_channels, _strides):
    # when `cardinality` == 1 this is just a standard convolution
    if cardinality == 1:
        return Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
    
    assert not nb_channels % cardinality
    _d = nb_channels // cardinality
    # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
    # and convolutions are separately performed within each group
    groups = []
    for j in range(cardinality):
        group = Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
        groups.append(Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
        
    # the grouped convolutional layer concatenates them as the outputs of the layer
    y = concatenate(groups)
    return y

def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
    """
    Our network consists of a stack of residual blocks. These blocks have the same topology,
    and are subject to two simple rules:
    - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
    - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
    """
    shortcut = y
    # we modify the residual building block as a bottleneck design to make the network more economical
    y = Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    y = add_common_layers(y)
    # ResNeXt (identical to ResNet when `cardinality` == 1)
    y = grouped_convolution(y, nb_channels_in, _strides=_strides)
    y = add_common_layers(y)
    y = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
    # batch normalization is employed after aggregating the transformations and before adding to the shortcut
    y = LayerNormalization()(y)
    # identity shortcuts used directly when the input and output are of the same dimensions
    if _project_shortcut or _strides != (1, 1):
        # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
        # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
        shortcut = Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
        shortcut = LayerNormalization()(shortcut)
    y = add([shortcut, y])
    # relu is performed right after each batch normalization,
    # expect for the output of the block where relu is performed after the adding to the shortcut
    y = LeakyReLU()(y)
    return y

def make_anime_discriminator(input_shape):
    model_in = Input(shape=input_shape, name='input')

    output = Conv2D(64, kernel_size=(5, 5), strides=3, input_shape=input_shape, padding="same")(model_in)
    output = LeakyReLU()(output)

    output = residual_block(output, 64, 128, (2, 2))
    output = residual_block(output, 128, 256, (2, 2))
    output = residual_block(output, 256, 512, (2, 2))
    output = residual_block(output, 512, 512, (2, 2))

    output = Flatten()(output)
    output = Dense(1, activation='linear', name='output')(output)

    return Model(inputs=model_in, outputs=output, name="discriminator")

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
              epochs=12,
              verbose=1,
              validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
