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

from data_prep import prepare_cifar10
from generator import make_small_generator
from discriminator import make_discriminator

import matplotlib.pyplot as plt

import os

import numpy as np

# parameters to tune
MAX_ITERATIONS = 10000
N_CRITIC = 5
BATCH_SIZE = 32
SAMPLE_INTERVAL = 50
IMAGE_SHAPE = (32, 32)
IMAGE_SHAPE_CH = (32, 32, 3)
LOG_FILE = 'logs/imp_wgan_dummy_logs.txt'
CRITIC_WEIGHTS_SAVE_LOC = 'weights/imp_wgan_dummy_critic.h5'
GENERATOR_WEIGHTS_SAVE_LOC = 'weights/imp_wgan_dummy_gen.h5'
# the below should be a folder
IMAGES_SAVE_DIR = "results"

class RandomWeightedAverage(_Merge):
    """Provides a (random) weighted average between real and generated image samples"""
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
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
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)

def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = generator.predict(noise)
    # Rescale images 0 - 1
    gen_imgs = 0.5 * (gen_imgs + 1)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,:])
            axs[i,j].axis('off')
            cnt += 1
    fig.savefig(os.path.join(IMAGES_SAVE_DIR, "cifar_%d.png" % epoch))
    plt.close()

def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

optimizer = Adam(0.0001, beta_1=0.5, beta_2=0.9)
# Build the generator and critic
generator = make_small_generator()
critic = make_discriminator(IMAGE_SHAPE_CH)

# Freeze generator's layers while training critic
generator.trainable = False
# Image input (real sample)
real_img = Input(shape=IMAGE_SHAPE_CH)
# Noise input
z_disc = Input(shape=(100,))
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
                  averaged_samples=interpolated_img)
partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function name

critic_model = Model(inputs=[real_img, z_disc],
                    outputs=[valid, fake, validity_interpolated])
critic_model.compile(loss=[wasserstein_loss,
                            wasserstein_loss,
                            partial_gp_loss],
                            optimizer=optimizer,
                            loss_weights=[1, 1, 10])

# For the generator we freeze the critic's layers
critic.trainable = False
generator.trainable = True
# Sampled noise for input to generator
z_gen = Input(shape=(100,))
# Generate images based of noise
img = generator(z_gen)
# Discriminator determines validity
valid = critic(img)
# Defines generator model
generator_model = Model(z_gen, valid)
generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)

# Load the dataset
datagen = prepare_cifar10(BATCH_SIZE)
# Adversarial ground truths
valid = -np.ones((BATCH_SIZE, 1))
fake =  np.ones((BATCH_SIZE, 1))
dummy = np.zeros((BATCH_SIZE, 1)) # Dummy gt for gradient penalty
for epoch in range(MAX_ITERATIONS):
    for _ in range(N_CRITIC):
        # get real images
        imgs = next(datagen)[0]
        # if we run out of data, generate more.
        if imgs.shape[0] != BATCH_SIZE:
            datagen = prepare_cifar10(BATCH_SIZE)
            imgs = next(datagen)[0]
        imgs = (imgs.astype(np.float32) - 0.5) * 2.0
        # Sample generator input
        noise = np.random.normal(0, 1, (BATCH_SIZE, 100))
        # Train the critic
        d_loss = critic_model.train_on_batch([imgs, noise],
                                                        [valid, fake, dummy])

    g_loss = generator_model.train_on_batch(noise, valid)

    # Plot the progress
    print ("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))
    # we also record it in a file
    with open(LOG_FILE, 'a+') as f:
        f.write('%d %f %f\n' % (epoch, d_loss[0], g_loss))


    # If at save interval => save generated image samples
    if epoch % SAMPLE_INTERVAL == 0:
        sample_images(epoch)
        critic.save_weights('weights/improved_up_critic.h5')
