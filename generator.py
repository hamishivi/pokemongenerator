'''
Improved generator, using a mixture of upsampling and
convolutional layers instead of transpose layers. I tested on a
semantic segmentation dataset, and so for that also add an encoder 
block that mirrors the decoder block (that forms the generator 
in the actual WGAN).
'''
import keras
from keras.models import Sequential, Model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import UpSampling2D, Activation, Conv2D, Dense, \
    BatchNormalization, Reshape, Input, MaxPooling2D
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU

def make_generator(input_shape=(100,), demo=False):
    model = Sequential()
    # encoder-decoder for testing
    # else just the decoder part
    if demo:
        model.add(Conv2D(3, kernel_size=3, input_shape=input_shape, 
                         data_format='channels_last', padding='same'))
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
        model.add(BatchNormalization())
        model.add(LeakyReLU())
        model.add(Reshape((4, 4, 512)))

        model.add(Conv2D(512, kernel_size=3, padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(256, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(32, kernel_size=3, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(UpSampling2D())
    model.add(Conv2D(3, kernel_size=3, padding='same', data_format='channels_last'))
    if __name__ == '__main__':
        model.add(Activation("softmax"))
    else:
        model.add(Activation('tanh', name='output'))

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
def ad20k(filepath):
    model = make_generator(input_shape=(128, 128, 3), demo=True)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
                  metrics=["accuracy"])
    # evaluate
    model.load_weights(filepath, by_name=False)
    test_datagen = get_demo_data('segmentation_dataset/images/test/')
    score = model.evaluate_generator(test_datagen, steps=14, verbose=1)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])


if __name__ == '__main__':
    generator = make_generator(input_shape=(128, 128, 3), demo=True)
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
    generator.evaluate_generator(test_datagen, steps=14, verbose=1)
