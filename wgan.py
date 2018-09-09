from discriminator import make_discriminator, compile_wasserstein_critic, EM_loss
from generator import make_generator
from data_prep import prepare_images

import sys

import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Input

import numpy as np

import matplotlib
# for server running
matplotlib.use('Agg')
from matplotlib import pyplot as plt

EPOCHS = 25000
N_CRITIC = 5
batch_size = 100
sample_interval = 100
image_shape = (128, 128, 3)

print("Welcome to the Pokemon WGAN!")
print("Generating images...")

datagen = prepare_images("data", batch_size, (128, 128))

print("Images ready. Making model...")

generator = make_generator()
discriminator = make_discriminator((128, 128, 3))
compile_wasserstein_critic(discriminator)

gen_in = Input(shape=(100,))
generated_img = generator(gen_in)

discriminator.trainable = False
is_valid = discriminator(generated_img)

combined = Model(gen_in, is_valid)
combined.compile(loss=EM_loss,
optimizer=keras.optimizers.RMSprop(lr=0.00005),
metrics=['accuracy'])

print("Models built! Starting to train...")

# labels for training
valid = -np.ones((batch_size, 1))
fake = np.ones((batch_size, 1))

for epoch in range(EPOCHS):
    # train discriminator
    d_iters = N_CRITIC
    if epoch < 25 or epoch % 500 == 0:
        d_iters = 100
    for _ in range(d_iters):
        # get real images
        imgs = next(datagen)[0]
        # if we run out of data, randomly generate more.
        if (imgs.shape[0] != batch_size):
            datagen = prepare_images("data", batch_size, (128, 128))
            imgs = next(datagen)[0]
        # rescale -1 to 1
        imgs = (imgs.astype(np.float32) - 127.5) / 127.5
        # get fake images from generator
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))
        # Generate a batch of new images
        fake_imgs = generator.predict(noise)
        
        # train!
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        # clip weights
        for layer in discriminator.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            layer.set_weights(weights)

        print(".", end="", flush=True)
            
    # train generator
    g_loss =combined.train_on_batch(noise, valid)

    print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

    if epoch % sample_interval == 0:
        r, c = 5, 5
        noise = np.random.normal(-1, 1, (batch_size, 100)).astype('float32')
        gen_imgs = generator.predict(noise, batch_size=5*5)

        gen_imgs = 0.5 * (gen_imgs + 1)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in axs:
            for p in i:
                p.imshow(gen_imgs[cnt, :,:,:])
                p.axis("off")
                cnt += 1
        fig.savefig("images/pokemon_" + str(epoch) + ".png")
