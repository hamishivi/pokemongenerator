'''
Main file that runs the vanilla WGAN training algorithm.
This was loosely based on
https://github.com/eriklindernoren/Keras-GAN/blob/master/wgan/wgan.py
but I have changed many elements of it.
'''
import os
import numpy as np
import matplotlib
# for server running
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import keras
from keras.models import Model
import keras.backend as K
from keras.layers import Input

# import models - change me to change models used
from discriminator import make_poke_discriminator
from generator import make_generator
from data_prep import prepare_images

def EM_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

# parameters to tune
MAX_ITERATIONS = 10000
N_CRITIC = 10
BATCH_SIZE = 64
SAMPLE_INTERVAL = 50
IMAGE_SHAPE = (128, 128)
IMAGE_SHAPE_CH = (128, 128, 3)
LOG_FILE = 'logs/dummy_logs.txt'
CRITIC_WEIGHTS_SAVE_LOC = 'weights/dummy_critic.hdf5'
GENERATOR_WEIGHTS_SAVE_LOC = 'weights/dummy_generator.hdf5'
# make sure this folder exists, or the training will error out.
IMAGES_SAVE_DIR = 'results'

print("Welcome to the Pokemon WGAN!")
print("Preparing images...")

# preprocess data
datagen = prepare_images("data", BATCH_SIZE, IMAGE_SHAPE)

print("Images prepped. Making WGAN...")

# make our models
generator = make_generator()
discriminator = make_poke_discriminator(IMAGE_SHAPE_CH)
discriminator.compile(loss=EM_loss,
                      optimizer=keras.optimizers.RMSprop(lr=0.00005),
                      metrics=["accuracy"])

gen_in = Input(shape=(100,))
generated_img = generator(gen_in)

discriminator.trainable = False
is_valid = discriminator(generated_img)

combined = Model(gen_in, is_valid)
combined.compile(loss=EM_loss,
                 optimizer=keras.optimizers.RMSprop(lr=0.00005),
                 metrics=['accuracy'])

print("WGAN built! Starting to train...")

# labels for training
valid = -np.ones((BATCH_SIZE, 1))
fake = np.ones((BATCH_SIZE, 1))

for epoch in range(MAX_ITERATIONS):
    d_iters = N_CRITIC
    # the second iteration of the wgan paper suggests doing this
    # to help the discriminator reach convergence faster.
    if epoch < 25 or epoch % 500 == 0:
        d_iters = 100
    for _ in range(d_iters):
        # get real images
        imgs = next(datagen)[0]
        # if we run out of data, generate more.
        if imgs.shape[0] != BATCH_SIZE:
            datagen = prepare_images("data", BATCH_SIZE, IMAGE_SHAPE)
            imgs = next(datagen)[0]

        # rescale images to range [-1, 1]
        imgs = (imgs.astype(np.float32) - 0.5) * 2.0
        # get fake images from generator
        # Sample noise as generator input to generate fake images
        noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100]).astype('float32')
        fake_imgs = generator.predict(noise)
        # train!
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = np.mean(0.5 * np.add(d_loss_real, d_loss_fake))
        # clip weights
        for layer in discriminator.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            layer.set_weights(weights)
        # print a dot so we can see the training is going along
        print(".", end="", flush=True)

    # train generator
    g_loss = np.mean(combined.train_on_batch(noise, valid))

    # end of iteration: record loss
    print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss, g_loss))
    # we also record it in a file
    with open(LOG_FILE, 'a+') as f:
        f.write('%d %f %f\n' % (epoch, d_loss, g_loss))

    if epoch % SAMPLE_INTERVAL == 0:
        noise = np.random.normal(-1, 1, (25, 100)).astype('float32')
        gen_imgs = generator.predict(noise, batch_size=25).astype('float32')

        gen_imgs = 0.5 * (gen_imgs + 1.0)
        gen_imgs = np.clip(gen_imgs, 0, 1)

        fig, axs = plt.subplots(5, 5)
        cnt = 0
        for i in axs:
            for p in i:
                p.imshow(gen_imgs[cnt, :, :, :])
                p.axis("off")
                cnt += 1
        fig.savefig(os.path.join(IMAGES_SAVE_DIR, "pokemon_" + str(epoch) + ".png"))
        fig.clear()
        # also save model at checkpoints
        discriminator.save(CRITIC_WEIGHTS_SAVE_LOC)
        generator.save(GENERATOR_WEIGHTS_SAVE_LOC)
