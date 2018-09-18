"""
An alternate discriminator implementation,
based off DenseNet
"""
import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import ZeroPadding2D, Dropout, Conv2D, Dense, Flatten, BatchNormalization, Input, AveragePooling2D, Concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.pooling import GlobalActivationPooling2D


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
    nb_layers = 5
    nb_filter = 5
    nb_dense_block = 4
    growth_rate = 1

    x = Conv2D(nb_filter, (3,3), padding="same")(x)

    for _ in range(nb_dense_block-1):
        x, nb_filter = dense_block(x, nb_filter, nb_layers, growth_rate)
        x = transition_layer(x, nb_filter)
    
    x, nb_filter = dense_block(x, nb_filter, nb_layers, growth_rate)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = GlobalActivationPooling2D()(x)
    x = Dense(1, activation='linear')(x)

    return Model(inputs=[model_input], outputs=[x], name="discriminator")

