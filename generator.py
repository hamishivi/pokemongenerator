import glob
import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, Input
from keras.layers.advanced_activations import LeakyReLU
from discriminator import make_discriminator
from data_prep import prepare_images
from scipy.misc import imsave, imread

def make_generator(input_shape=(100,)):
    model = Sequential()
    if __name__ == "__main__":
        model.add(Conv2D(2, kernel_size=(5, 5), strides=[4,4], input_shape=input_shape, data_format="channels_last", padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2D(64, kernel_size=(5, 5), strides=[2,2], input_shape=input_shape, data_format="channels_last", padding='same'))
        model.add(LeakyReLU())

        model.add(Conv2D(128, kernel_size=(5, 5), strides=[2,2], input_shape=input_shape, data_format="channels_last", padding='same'))
        model.add(LeakyReLU())

    else:
        # takes in 100dim noise vector as seed
        model.add(Dense(4 * 4 * 512, input_dim=100))
    model.add(LeakyReLU())

    model.summary()

    model.add(Reshape((4, 4, 512)))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(256, kernel_size=(5, 5), strides=[4,4], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2D(128, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(64, kernel_size=(5, 5), strides=[4,4], padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU())

    model.add(Conv2DTranspose(3, kernel_size=(5, 5), strides=[2,2], padding='same', data_format='channels_last'))
    model.add(Activation("tanh"))

    noise = Input(shape=input_shape)
    
    img = model(noise)

    model.summary()

    return Model(noise, img)

def get_demo_data():
    # we create two instances with the same arguments
    data_gen_args = dict(rotation_range=90.,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         zoom_range=0.2)
    image_datagen = ImageDataGenerator(**data_gen_args)
    mask_datagen = ImageDataGenerator(**data_gen_args)
    
    # Provide the same seed and keyword arguments to the fit and flow methods
    seed = 1
    
    image_generator = image_datagen.flow_from_directory(
        'segmentation_dataset/images/raws',
        target_size=(128,128),
        class_mode=None,
        seed=seed)
    
    mask_generator = mask_datagen.flow_from_directory(
        'segmentation_dataset/images/masks',
        target_size=(128,128),
        class_mode=None,
        seed=seed)
    
    # combine generators into one which yields image and masks
    return zip(image_generator, mask_generator)

def get_demo_test():
    test_datagen = ImageDataGenerator()
    image_generator = test_datagen.flow_from_directory(
        'segmentation_dataset/images/test',
        class_mode=None)
    return image_generator

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
        optimizer=keras.optimizers.Adam(lr=1e-5),
        metrics=["accuracy"])
    train_datagen = get_demo_data()
    deconv_layers.fit_generator(train_datagen, steps_per_epoch=1000, epochs=1)

    test_datagen = get_demo_test()
    print("Showing example segmentation...")
    image = imread('segmentation_dataset/images/test/abbey/ADE_val_00000001.jpg')
    image = np.expand_dims(image, axis=0)
    print(image.shape)
    results = deconv_layers.predict(image, test_datagen, verbose=1)
    print(results.shape)
    for idx, image in enumerate(results):
        imsave("result_." + str(idx) + "png", image)
