'''
This is the training code for the WGAN-GP.
This is largely copied from https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan_gp/wgan_gp.py
I have changed the optimiser, the architectures of the generator and discriminator, the dataset used,
and otherwise streamlined the code somewhat and made it fit with my setup.
'''

from keras.layers.merge import _Merge
from keras.layers import Input
from keras.models import  Model
from keras.optimizers import Adam
from functools import partial

import keras.backend as K

from data_prep import prepare_images, prepare_mnist, prepare_cifar10, prepare_anime_images
from alt_gen import make_mnist_generator, make_cifar_generator, make_anime_generator
from generator import make_generator
from discriminator import make_discriminator, make_poke_discriminator

import matplotlib.pyplot as plt

import os
import sys

import numpy as np

# parameters to tune
MAX_ITERATIONS = 100000
N_CRITIC = 5
BATCH_SIZE = 64
SAMPLE_INTERVAL = 50
LOG_FILE = 'logs/dummy_logs.txt'
CRITIC_WEIGHTS_SAVE_LOC = 'weights/imp_wgan_dummy_critic.h5'
GENERATOR_WEIGHTS_SAVE_LOC = 'weights/imp_wgan_dummy_gen.h5'
# the below should be a folder
IMAGES_SAVE_DIR = "results"
GRADIENT_PENALTY_WEIGHT = 10



mode = 'pokemon'
if len(sys.argv) > 1:
    if sys.argv[1] == 'mnist':
        mode = 'mnist'
    elif sys.argv[1] == 'cifar':
        mode = 'cifar'
    elif sys.argv[1] == 'anime':
        mode = 'anime'
    elif sys.argv[1] == 'pokemon-alt':
        mode = 'alt'

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((BATCH_SIZE, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

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

def sample_images(g_model, epoch, noise, image_save_dir, greyscale=False):
    r, c = 5, 5
    gen_imgs = g_model.predict(noise)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * (gen_imgs + 1)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            if greyscale:
                axs[i,j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            else:
                axs[i,j].imshow(gen_imgs[cnt, :, :, :])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(image_save_dir, "image_%d.png" % epoch))
    plt.close()

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def mean_loss(y_true, y_pred):
    return K.mean(y_pred)

optimizer = Adam(0.0005, beta_1=0.0, beta_2=0.9999)
input_dim = 100
# Build the generator and critic
generator = make_generator()
image_shape = (128, 128, 3)
if mode == 'mnist':
    generator = make_mnist_generator()
    image_shape = (28, 28, 1)
elif mode == 'cifar':
    generator = make_cifar_generator()
    image_shape = (32, 32, 3)
elif mode == 'anime':
    generator = make_anime_generator()
    image_shape = (48, 48, 3)
    input_dim = 40
elif mode == 'alt':
    image_shape = (48, 48, 3)
    generator = make_anime_generator()
    input_dim = 40

CONST_NOISE = np.random.normal(0, 1, (25, input_dim))

# we currently use the same discriminator across all
critic = make_discriminator(image_shape, batchnorm=False)
if mode == 'pokemon':
    optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
    critic = make_poke_discriminator(image_shape, batchnorm=False)

# Freeze generator's layers while training critic
generator.trainable = False
for l in generator.layers:
    l.trainable = False
# Image input (real sample)
real_img = Input(shape=image_shape)
# Noise input
z_disc = Input(shape=(input_dim,))
# Generate image based of noise (fake sample)
fake_img = generator(z_disc)
# Discriminator determines validity of the real and fake images
fake = critic(fake_img)
valid = critic(real_img)
# Construct weighted average between real and fake images
interpolated_img = RandomWeightedAverage()([real_img, fake_img])
# Determine validity of weighted sample
validity_interpolated = critic(interpolated_img)

# Use Python partial to provide loss function with additional
# 'averaged_samples' argument
partial_gp_loss = partial(gradient_penalty_loss,
                  averaged_samples=interpolated_img,
                  gradient_penalty_weight=GRADIENT_PENALTY_WEIGHT)
partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function name

critic_model = Model(inputs=[real_img, z_disc],
                    outputs=[valid, fake, validity_interpolated])
critic_model.compile(loss=[wasserstein_loss,
                            wasserstein_loss,
                            partial_gp_loss],
                            optimizer=optimizer)

# For the generator we freeze the critic's layers
critic.trainable = False
for l in critic.layers:
    l.trainable = False
generator.trainable = True
for l in generator.layers:
    l.trainable = True
# Sampled noise for input to generator
z_gen = Input(shape=(input_dim,))
# Generate images based of noise
img = generator(z_gen)
# Discriminator determines validity
gen_crit_out = critic(img)
# Defines generator model
generator_model = Model(inputs=[z_gen], outputs=[gen_crit_out])
generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

with open(LOG_FILE, 'w') as f:
    f.write("")

# Load the dataset
prepare_function = lambda x: prepare_images('data', x, (image_shape[0], image_shape[1])) # TODO fix this!
if mode == 'mnist':
    prepare_function = prepare_mnist
elif mode == 'cifar':
    prepare_function = prepare_cifar10
elif mode == 'anime':
    prepare_function = lambda x: prepare_anime_images('data', x, (image_shape[0], image_shape[1]))

datagen = prepare_function(BATCH_SIZE)
# Adversarial ground truths
valid = np.ones((BATCH_SIZE, 1), dtype=np.float32)
fake = -valid
dummy = -np.zeros((BATCH_SIZE, 1), dtype=np.float32) # Dummy gt for gradient penalty

a_epoch = 0
for epoch in range(0, MAX_ITERATIONS+1):
    for _ in range(N_CRITIC):
        # get real images
        try:
            imgs = next(datagen)[0]
        except StopIteration:
            datagen = prepare_function(BATCH_SIZE)
            imgs = next(datagen)[0]
        # if we run out of data, generate more.
        if imgs.shape[0] != BATCH_SIZE:
            datagen = prepare_function(BATCH_SIZE)
            imgs = next(datagen)[0]

        imgs = (imgs.astype(np.float32) - 0.5) * 2.0
        # Sample generator input
        noise =  np.random.normal(0, 1, (BATCH_SIZE, input_dim))
        # Train the critic
        d_loss = critic_model.train_on_batch([imgs, noise],
                                                        [valid, fake, dummy])

    g_loss = generator_model.train_on_batch(np.random.normal(0, 1, (BATCH_SIZE, input_dim)), valid)

    # Plot the progress
    print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
    # we also record it in a file
    with open(LOG_FILE, 'a+') as f:
        f.write('%d %f %f\n' % (epoch, d_loss[0], g_loss))


    # If at save interval => save generated image samples
    if epoch % SAMPLE_INTERVAL == 0:
        sample_images(generator, epoch, CONST_NOISE, IMAGES_SAVE_DIR, greyscale=(mode == 'mnist'))
        critic.save_weights(CRITIC_WEIGHTS_SAVE_LOC)
        generator.save_weights(GENERATOR_WEIGHTS_SAVE_LOC)

sample_images(generator, MAX_ITERATIONS, CONST_NOISE, IMAGES_SAVE_DIR, greyscale=(mode == 'mnist'))
critic.save_weights(CRITIC_WEIGHTS_SAVE_LOC)
generator.save_weights(GENERATOR_WEIGHTS_SAVE_LOC)