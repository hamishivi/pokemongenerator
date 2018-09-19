"""
An alternate discriminator implementation,
based off DenseNet
"""
import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import ZeroPadding2D, Dropout, Conv2D, Dense, Flatten, BatchNormalization, Input, AveragePooling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalAveragePooling2D


# we feed +1 as label for real and -1 for fake images
# in the D, and opposite in the G.
def EM_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def compile_wasserstein_critic(model):
    model.compile(loss=EM_loss,
        optimizer=keras.optimizers.RMSprop(lr=0.00005),
        metrics=["accuracy"])

def conv_block(x, nb_filter):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(nb_filter, (3,3), padding="same")(x)
    x = Dropout(0.25)(x)
    return x

def transition_layer(x, nb_filter):
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Conv2D(nb_filter, (1,1), padding="same")(x)
    x = Dropout(0.25)(x)
    x = AveragePooling2D((2,2), strides=(2,2))(x)
    return x

def dense_block(x, nb_filter, nb_layers, growth_rate):
    feature_list = [x]
    for _ in range(nb_layers):
        x = conv_block(x, nb_filter)
        feature_list.append(x)
        x = Concatenate()(feature_list)
        nb_filter += growth_rate
    return x, nb_filter

def make_discriminator(input_shape):
    model_input = Input(shape=input_shape)

    # parameters for the densenet
    nb_layers = 4
    nb_filter = 16
    nb_dense_block = 2
    growth_rate = 12

    x = Conv2D(nb_filter, (3,3), padding="same")(model_input)

    for _ in range(nb_dense_block-1):
        x, nb_filter = dense_block(x, nb_filter, nb_layers, growth_rate)
        x = transition_layer(x, nb_filter)
    
    x, nb_filter = dense_block(x, nb_filter, nb_layers, growth_rate)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = GlobalAveragePooling2D()(x)
    if __name__ == "__main__":
        x = Dense(10, activation='softmax')(x)
    else:
        x = Dense(1, activation='linear')(x)

    return Model(inputs=[model_input], outputs=[x], name="discriminator")

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
