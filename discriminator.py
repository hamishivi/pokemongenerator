import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import ZeroPadding2D, Dropout, Conv2D, Dense, Flatten, BatchNormalization, Input
from keras.layers.advanced_activations import LeakyReLU


# we feed +1 as label for real and -1 for fake images
# in the D, and opposite in the G.
def EM_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def make_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, input_shape=input_shape, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(256, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Conv2D(512, kernel_size=3, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())

    if __name__ == "__main__":
        model.add(Dense(10, activation='softmax'))
    else:
        model.add(Dense(1, activation='linear'))

    img = Input(shape=input_shape)
    validity = model(img)

    return Model(img, validity, name="discriminator")

def compile_wasserstein_critic(model):
    model.compile(loss=EM_loss,
        optimizer=keras.optimizers.RMSprop(lr=0.00005),
        metrics=["accuracy"])

def compile_demo(model):
    model.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.RMSprop(lr=0.00005),
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
