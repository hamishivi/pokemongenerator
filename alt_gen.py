"""
A alternate generator setup, using transpose layers instead of
upsampling. I tested on a semantic segmentation dataset, and so
for that also add an encoder block that mirrors the decoder block
(that forms the generator in the actual WGAN). This was the baseline
generator model used in the report.
"""
import keras
from keras.utils import plot_model
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, \
    BatchNormalization, Reshape, Input, MaxPooling2D
from keras.layers.advanced_activations import LeakyReLU

def make_alt_generator(input_shape=(100,), demo=False):
    model = Sequential()

    if demo:
        model.add(Conv2D(3, kernel_size=3, input_shape=input_shape,
                         data_format="channels_last", padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling2D())

        model.add(Conv2D(32, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling2D())

        model.add(Conv2D(64, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling2D())

        model.add(Conv2D(128, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling2D())

        model.add(Conv2D(256, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(MaxPooling2D())
    else:
        # takes in 100dim noise vector as seed
        model.add(Dense(4 * 4 * 512, input_dim=100, name='input'))
        model.add(Reshape((4, 4, 512)))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(32, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(16, kernel_size=5, strides=[2,2], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size=5, padding='same'))
    if demo:
        model.add(Activation("softmax"))
    else:
        model.add(Activation('tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

def make_mnist_generator(input_shape=(100,)):
    '''For the MNIST digit generation'''
    model = Sequential()

    model.add(Dense(256 * 7 * 7, input_dim=100, name='input'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((7, 7, 256)))

    model.add(Conv2DTranspose(256, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(128, (5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, (5, 5), strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(1, (5, 5), padding='same', activation='tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

def make_anime_generator(input_shape=(40,)):
    '''For the anime face generation'''
    model = Sequential()

    model.add(Dense(1024 * 1 * 1, input_dim=40, name='input'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((1, 1, 1024)))

    model.add(Conv2DTranspose(1024, kernel_size=3, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size=4, padding='same', strides=2, activation='tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

def make_cifar_generator(input_shape=(100,)):
    '''For the cifar digit generation'''
    model = Sequential()

    model.add(Dense(1024 * 1 * 1, input_dim=100, name='input'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())
    model.add(Reshape((1, 1, 1024)))

    model.add(Conv2DTranspose(1024, kernel_size=2, strides=1, padding='valid'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(512, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(128, kernel_size=4, strides=2, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size=4, padding='same', strides=2, activation='tanh', name='output'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

def get_demo_data(directory):
    # we create two instances with the same arguments
    image_datagen = ImageDataGenerator()
    mask_datagen = ImageDataGenerator()

    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1

    image_generator = image_datagen.flow_from_directory(
        directory,
        classes=['raws'],
        target_size=(128, 128),
        class_mode=None,
        batch_size=128,
        seed=seed)

    mask_generator = mask_datagen.flow_from_directory(
        directory,
        classes=['masks'],
        target_size=(128, 128),
        class_mode=None,
        batch_size=128,
        seed=seed)

    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)

# for pretrained model
def ad20k_alt(filepath):
    model = make_alt_generator(input_shape=(128, 128, 3), demo=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                  metrics=["accuracy"])
    model.load_weights(filepath, by_name=False)
    # show example
    test_example = next(get_demo_data('segmentation_dataset/images/test/'))
    raw = test_example[0][0]
    # normalise down to [0,1] range
    raw = raw/255.0
    return raw, model.predict(test_example[0], batch_size=128)[0]


if __name__ == '__main__':
    generator = make_alt_generator(input_shape=(128, 128, 3), demo=True)
    generator.compile(loss=keras.losses.categorical_crossentropy,
                      optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                      metrics=["accuracy"])
    train_datagen = get_demo_data('segmentation_dataset/images/')
    # train
    generator.fit_generator(train_datagen,
                            steps_per_epoch=122,
                            epochs=50,
                            validation_data=get_demo_data('segmentation_dataset/images/test/'),
                            validation_steps=14)
    # evaluate
    test_datagen = get_demo_data('segmentation_dataset/images/test/')
    score = generator.evaluate_generator(test_datagen, steps=14, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
