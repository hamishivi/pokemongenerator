from discriminator import make_discriminator, compile_wasserstein_critic, EM_loss
from generator import make_generator
from data_prep import prepare_images

import keras
from keras.models import Sequential, Model
import keras.backend as K
from keras.layers import Input

import matplotlib.pyplot as plt
import numpy as np

EPOCHS = 12
N_CRITIC = 5
batch_size = 64
sample_interval = 50
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
combined.compile(loss=EM_loss, optimizer=keras.optimizers.RMSprop(lr=0.00005, clipvalue=0.01), metrics=['accuracy'])

print("Models built! Starting to train...")

# labels for training
valid = -np.ones((batch_size, 1))
fake = np.ones((batch_size, 1))

for epoch in range(EPOCHS):
    # train discriminator
    for _ in range(N_CRITIC):
        # get real images
        imgs = next(datagen)[0]
        # get fake images from generator
        # Sample noise as generator input
        noise = np.random.normal(0, 1, (batch_size, 100))
        # Generate a batch of new images
        fake_imgs = generator.predict(noise, batch_size=batch_size)
        print("fake", fake_imgs.shape)
        # train!
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(fake_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
        
    # train generator
    g_loss = combined.train_on_batch(noise, valid)

    print ("%d [D loss: %f] [G loss: %f]" % (epoch, 1 - d_loss[0], 1 - g_loss[0]))

    if epoch % sample_interval == 0:
        r, c = 5, 5
        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise, batch_size=batch_size)

        gen_imgs = 0.5 * gen_imgs + 1

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/pokemon_%d.png" % epoch)
        plt.close()
       
