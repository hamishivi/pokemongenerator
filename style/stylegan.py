import numpy as np
from functools import partial
import sys
from PIL import Image
import os
import matplotlib
from matplotlib import pyplot as plt

from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
from keras.layers.merge import _Merge
import keras.backend as K

from AdaIN import AdaInstanceNormalization
from style_mapper import make_mapper
from style_gen import make_generator
from style_disc import make_discriminator
from data_prep import prepare_images, prepare_mnist, prepare_cifar10, prepare_anime_images

# constants
MAX_ITERATIONS = 100000
BATCH_SIZE = 32
SAMPLE_INTERVAL = 100
LOG_FILE = 'logs/dummy_logs.txt'
CRITIC_WEIGHTS_SAVE_LOC = 'weights/imp_wgan_dummy_critic.h5'
GENERATOR_WEIGHTS_SAVE_LOC = 'weights/imp_wgan_dummy_gen.h5'
# the below should be a folder
IMAGES_SAVE_DIR = "Results"

# determine which dataset we are using
mode = 'pokemon'
image_shape = (256, 256, 3)
if len(sys.argv) > 1:
    if sys.argv[1] == 'mnist':
        mode = 'mnist'
        image_shape = (28, 28, 1)
    elif sys.argv[1] == 'cifar':
        mode = 'cifar'
        image_shape = (32, 32, 3)
    elif sys.argv[1] == 'anime':
        mode = 'anime'
        image_shape = (48, 48, 3)
    elif sys.argv[1] == 'pokemon-alt':
        mode = 'alt'
        image_shape = (48, 48, 3)

# GP part of WGAN-GP training algorithm
# from https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
def gradient_penalty_loss(y_true, y_pred, averaged_samples, gradient_penalty_weight):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                              axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = gradient_penalty_weight * K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

im_size = image_shape[0]
latent_size = 512
lr = 0.0001

discriminator = make_discriminator(im_size)
generator = make_generator(im_size, latent_size)

# first model, AdModel: discriminator frozen, training generator and mapping network      
#D does not update
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False

#G does update
generator.trainable = True
for layer in generator.layers:
    layer.trainable = True

#This model is simple sequential one with inputs and outputs
gi = Input(shape = [latent_size])
gi2 = Input(shape = [im_size, im_size, 1])
gi3 = Input(shape = [1])

gf = generator([gi, gi2, gi3])
df = discriminator(gf)

AM = Model(inputs = [gi, gi2, gi3], outputs = df)

AM.compile(optimizer = Adam(lr, beta_1 = 0, beta_2 = 0.99), loss = wasserstein_loss)

# Model 3: DisModel, just training the discriminator
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True

generator.trainable = False
for layer in generator.layers:
    layer.trainable = False
        
# pass in a real image
ri = Input(shape = [im_size, im_size, 3])
dr = discriminator(ri)

# pass in a fake image (by feeding into generator first)
gi = Input(shape = [latent_size])
gi2 = Input(shape = [im_size, im_size, 1])
gi3 = Input(shape = [1])
gf = generator([gi, gi2, gi3])
df = discriminator(gf)

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

# Construct weighted average between real and fake images
interpolated_img = RandomWeightedAverage()([ri, gf])
# Determine validity of weighted sample
validity_interpolated = discriminator(interpolated_img)

DM = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, validity_interpolated])
# special GP Loss
partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, gradient_penalty_weight = 10)

# we have mse, mse, then gp loss
DM.compile(optimizer=Adam(lr, beta_1 = 0, beta_2 = 0.99), loss=[wasserstein_loss, wasserstein_loss, partial_gp_loss])

# okay, finally we have all the models we are using defined.
# lets load our data
# Load the dataset
prepare_function = lambda x: prepare_images('data', x, (image_shape[0], image_shape[1]))
if mode == 'mnist':
    prepare_function = prepare_mnist
elif mode == 'cifar':
    prepare_function = prepare_cifar10
elif mode == 'anime':
    prepare_function = lambda x: prepare_anime_images('data', x, (image_shape[0], image_shape[1]))

datagen = prepare_function(BATCH_SIZE)
# Adversarial ground truths
ones = np.ones((BATCH_SIZE, 1), dtype=np.float32)
nones = -ones
zeroes = np.zeros((BATCH_SIZE, 1), dtype=np.float32) # Dummy gt for gradient penalty

enoise = np.random.normal(0.0, 1.0, size = [8, latent_size])
enoiseImage = np.random.uniform(0.0, 1.0, size = [8, im_size, im_size, 1])

# train!    
# we alternate between mixed and regular models
for step in range(MAX_ITERATIONS+1):
    try:
        imgs = next(datagen)[0]
        if imgs.shape[0] < BATCH_SIZE:
            raise StopIteration  # throw out the last batch if it is less than batch size.
    except StopIteration:
        datagen = prepare_function(BATCH_SIZE)
        imgs = next(datagen)[0]
    # train discriminator
    train_noise = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size])
    train_noise_image = np.random.uniform(0.0, 1.0, size = [BATCH_SIZE, im_size, im_size, 1])
    d_loss = DM.train_on_batch(
        [imgs, train_noise, train_noise_image, ones],
        [ones, nones, ones]
    )
    # train generator
    train_noise = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size])
    train_noise_image = np.random.uniform(0.0, 1.0, size = [BATCH_SIZE, im_size, im_size, 1])
    g_loss = AM.train_on_batch([train_noise, train_noise_image, ones], ones)
        
    if step % SAMPLE_INTERVAL == 0:
        # todo, sample our images
        print(f"Step {step}:")
        print(f"D_loss: {str(d_loss)}")
        print("G_loss: " + str(g_loss))

        #evalTrunc(step)
        sample_noise = np.clip(np.random.normal(0.0, 1.0, size = [25, latent_size]), -1.8, 1.8)
        sample_noise_image = np.random.uniform(0.0, 1.0, size = [25, im_size, im_size, 1])
        
        gen_imgs = generator.predict([sample_noise, sample_noise_image, np.ones([25, 1])])

        #gen_imgs = 0.5 * (gen_imgs + 1.0)
        #gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in axs:
            for p in i:
                p.imshow(gen_imgs[cnt, :, :, :])
                p.axis("off")
                cnt += 1
        fig.savefig(os.path.join(IMAGES_SAVE_DIR, "pokemon_" + str(step) + ".png"))
        fig.clear()
