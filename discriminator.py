import keras
from keras.models import Sequential
import keras.backend as K
from keras.layers import Conv2D, Dense, Flatten, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


# we feed +1 as label for real and -1 for fake images
# in the D, and opposite in the G.
def EM_loss(y_true, y_pred):
    return K.mean(y_pred * y_true)

def make_discriminator(input_shape):
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(5, 5), input_shape=input_shape, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(64, kernel_size=(5, 5)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(LeakyReLU())
    model.add(Dense(1))
    return model

def compile_wasserstein_critic(model):
    model.compile(loss=EM_loss, optimizer=keras.optimizers.RMSprop(lr=0.00005, clipvalue=0.01), metrics=["accuracy"])

# Example: mnist data set (digit recognition)
if __name__ == "__main__":
    from keras.datasets import mnist
    import sys

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    y_train = [1 if y > 4 else -1 for y in y_train]
    y_test = [1 if y > 4 else -1 for y in y_test]
    x_train = (x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32')  - 127.5) / 127.5
    x_test = (x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32')  - 127.5) / 127.5
    model = make_discriminator((28, 28, 1))
    # no EM as that needs to be paired with the generator
    compile_wasserstein_critic(model)
    model.fit(x_train, y_train,
          batch_size=128,
          epochs=6,
          verbose=1,
          validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
