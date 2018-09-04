import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Activation, Conv2D, Conv2DTranspose, Dense, Flatten, BatchNormalization, Reshape, Input
from keras.layers.advanced_activations import LeakyReLU
from discriminator import make_discriminator
from data_prep import prepare_images
from PIL import Image

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
    batch_size = 64

    image_datagen = prepare_images("Semantic_dataset100/image", batch_size, (128, 128), shuffle=False, class_mode=None)
    mask_datagen = prepare_images("Semantic_dataset100/ground-truth", batch_size, (128, 128), shuffle=False, class_mode=None)

    test_datagen = prepare_images("Semantic_dataset100/test", batch_size, (128, 128))

    train_datagen = zip(image_datagen, mask_datagen)

    deconv_layers.fit_generator(train_datagen, steps_per_epoch=2000, epochs=5)

    print("Showing example segmentation...")
    results = deconv_layers.predict_generator(test_datagen,30,verbose=1)
    for idx, image in enumerate(results):
        im = Image.fromarray(image)
        im.save("result_." + idx + "png")
