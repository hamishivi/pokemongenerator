import os
import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import UpSampling2D, Activation, Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, Input, MaxPooling2D, Dropout

from keras.layers.advanced_activations import LeakyReLU
from discriminator import make_discriminator
from data_prep import prepare_images
from scipy.misc import imsave, imread

def make_generator(input_shape=(100,)):
    model = Sequential()
    # segnet encoder-decoder for testing
    # else just the decoder part
    if __name__ == "__main__":
        model.add(Conv2D(3, kernel_size=3, input_shape=input_shape, data_format="channels_last", padding='same'))
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
        model.add(Dense(4 * 4 * 512, input_dim=100))
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
        model.add(Activation('tanh'))

    noise = Input(shape=input_shape)
    img = model(noise)

    return Model(noise, img)

def get_demo_data():
    # we create two instances with the same arguments
    data_gen_args = dict()
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory(
        'segmentation_dataset/images/',
        classes=['raws'],
        target_size=(128,128),
        class_mode=None,
        batch_size=128,
        seed=seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        'segmentation_dataset/images/',
        classes=['masks'],
        target_size=(128,128),
        class_mode=None,
        batch_size=128,
        seed=seed)
    
    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)

def get_demo_test():
        # we create two instances with the same arguments
    data_gen_args = dict()
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory(
        'segmentation_dataset/images/test/',
        classes=['raws'],
        target_size=(128,128),
        class_mode=None,
        batch_size=128,
        seed=seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        'segmentation_dataset/images/test/',
        classes=['masks'],
        target_size=(128,128),
        class_mode=None,
        batch_size=128,
        seed=seed)
    
    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)

def compile_demo(deconv, conv):
    in_conv = Input(shape=(128,128,3))
    out_conv = conv(in_conv)
    out_deconv = deconv(out_conv)

    combined = Model(in_conv, out_deconv)

    combined.summary()

    combined.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adam(lr=1e-5),
        metrics=["accuracy"])
    return combined

if __name__ == '__main__':
    deconv_layers = make_generator(input_shape=(128,128,3))
    deconv_layers.compile(loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.SGD(lr=0.001, momentum=0.9),
        metrics=["accuracy"])
    train_datagen = get_demo_data()
    # train
    deconv_layers.fit_generator(train_datagen, steps_per_epoch=122, epochs=50, validation_data=get_demo_test(), validation_steps=14)
    # evaluate
    test_datagen = get_demo_test()
    deconv_layers.evaluate_generator(test_datagen, steps=14, verbose=1)
    # reset test datagen to see results
    test_datagen = get_demo_test()
    predict_images = next(test_datagen)
    test_raws = predict_images[1]
    results = deconv_layers.predict(predict_images[0], verbose=1)

    for idx, image in enumerate(results):
        imsave("seg_results/result_" + str(idx) + ".png", image)
        imsave("seg_results/image_" + str(idx) + ".png", test_raws[idx])

    
