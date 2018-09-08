from discriminator import make_discriminator, compile_wasserstein_critic, EM_loss
from generator import make_generator
from data_prep import prepare_images

import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Input

import numpy as np

import matplotlib
# for server running
matplotlib.use('Agg')
from matplotlib import pyplot as plt

EPOCHS = 4000
N_CRITIC = 5
batch_size = 32
sample_interval = 1
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
is_valid = discriminator(generated_img)

combined = Model(gen_in, is_valid)
combined.get_layer("discriminator").trainable = False
combined.compile(loss=EM_loss, optimizer=keras.optimizers.RMSprop(lr=0.00005), metrics=['accuracy'])

print("Models built! Starting to train...")

# labels for training
valid = -np.ones((batch_size, 1))
fake = np.ones((batch_size, 1))

for epoch in range(EPOCHS):
    # train discriminator
    discriminator.trainable = True
    for l in discriminator.layers: l.trainable = True
    if epoch < 25:
        N_CRITIC = 100
    else:
        N_CRITIC = 5
    for _ in range(N_CRITIC):
        # clip weights
        for layer in discriminator.layers:
            weights = layer.get_weights()
            weights = [np.clip(w, -0.01, 0.01) for w in weights]
            layer.set_weights(weights)
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
        noise = np.random.uniform(-1, 1, (batch_size, 100)).astype('float32')
        # Generate a batch of new images
        fake_imgs = generator.predict(noise, batch_size=batch_size)
        print(".", end="", flush=True)
        # train!
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = np.mean(d_loss_real + d_loss_fake)
            # train generator

    discriminator.trainable = False
    for l in discriminator.layers: l.trainable = False
    g_loss = np.mean(combined.train_on_batch(noise, valid))

    print ("\n%d [D loss: %f (real: %f, fake: %f)] [G loss: %f]" % (epoch, 1 - d_loss, d_loss_real, d_loss_fake, 1 - g_loss))

    if epoch % sample_interval == 0:
        r, c = 5, 5
        noise = np.random.normal(-1, 1, (batch_size, 100)).astype('float32')
        gen_imgs = generator.predict(noise, batch_size=batch_size)

        gen_imgs = 0.5 * (gen_imgs + 1)
        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in axs:
            for p in i:
                p.imshow(gen_imgs[0, :,:,:])
                p.axis("off")
                cnt += 1
        fig.savefig("images/pokemon_" + str(epoch) + ".png")

