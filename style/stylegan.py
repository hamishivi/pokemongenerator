import numpy as np
from functools import partial
import sys
from PIL import Image

from keras.layers import Conv2D, Dense, AveragePooling2D, LeakyReLU, Activation
from keras.layers import Reshape, UpSampling2D, Dropout, Flatten, Input, add, Cropping2D
from keras.models import model_from_json, Model
from keras.optimizers import Adam
import keras.backend as K

from AdaIN import AdaInstanceNormalization
from .style_mapper import make_mapper
from style_gen import make_generator
from style_disc import make_discriminator
from ..data_prep import prepare_images, prepare_mnist, prepare_cifar10, prepare_anime_images

# constants
BATCH_SIZE = 32
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

# determine which dataset we are using
mode = 'pokemon'
image_shape = (128, 128, 3)
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

im_size = 100
latent_size = 100
lr = 0.01

discriminator = make_discriminator(im_size)
generator, style_layers = make_generator(im_size)
mapper = make_mapper(latent_size)

# first model, AdModel: discriminator frozen, training generator and mapping network      
#D does not update
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False

#G does update
generator.trainable = True
for layer in generator.layers:
    layer.trainable = True

#S does update
mapper.trainable = True
for layer in mapper.layers:
    layer.trainable = True

#This model is simple sequential one with inputs and outputs
gi = Input(shape = [latent_size])
gs = mapper(gi)
gi2 = Input(shape = [im_size, im_size, 1])
gi3 = Input(shape = [1])

gf = generator(([gs] * style_layers) + [gi2, gi3])
df = discriminator(gf)

AM = Model(inputs = [gi, gi2, gi3], outputs = df)

AM.compile(optimizer = 'adam', loss = 'mse')

# second model, mixedModel: again, freezing discriminator and training generator and mapper -
# but we also introduce some regularities
discriminator.trainable = False
for layer in discriminator.layers:
    layer.trainable = False
        
generator.trainable = True
for layer in generator.layers:
    layer.trainable = True
            
mapper.trainable = True
for layer in mapper.layers:
    layer.trainable = True
        
# generator style inputs (noise and feature vector)
inp_s = [Input([latent_size]) for _ in range(style_layers)]
ss = [mapper(inp_s[i]) for i in range(style_layers)]
# + two inputs for   
gi2 = Input(shape = [im_size, im_size, 1])
gi3 = Input(shape = [1])
# feed into generator and then discriminator
gf = generator(ss + [gi2, gi3])
df = discriminator(gf)
# compile!
MM = Model(inputs = inp_s + [gi2, gi3], outputs = df)
MM.compile(optimizer = 'adam', loss = 'mse')

# Model 3: DisModel, just training the discriminator
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True

generator.trainable = False
for layer in generator.layers:
    layer.trainable = False

mapper.trainable = False
for layer in mapper.layers:
    layer.trainable = False
        
# pass in a real image
ri = Input(shape = [im_size, im_size, 3])
dr = discriminator(ri)

# pass in a fake image (by feeding into generator first)
gi = Input(shape = [latent_size])
gs = mapper(gi)
gi2 = Input(shape = [im_size, im_size, 1])
gi3 = Input(shape = [1])
gf = generator(([gs] * style_layers) + [gi2, gi3])
df = discriminator(gf)

DM = Model(inputs=[ri, gi, gi2, gi3], outputs=[dr, df, dr])
# special GP Loss
partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 50)

# we have mse, mse, then gp loss
DM.compile(optimizer=Adam(lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])

# final model! the mixed/regularised version of the DisModel, MixModelD
discriminator.trainable = True
for layer in discriminator.layers:
    layer.trainable = True

#G does not update
generator.trainable = False
for layer in generator.layers:
    layer.trainable = False

#S does not update
mapper.trainable = False
for layer in mapper.layers:
    layer.trainable = False
        
# style inputs (noise and features)
inp_s = [Input([latent_size]) for _ in range(style_layers)]
ss = [mapper(inp_s[i]) for i in range(style_layers)]
# two additional inputs
gi2 = Input(shape = [im_size, im_size, 1])
gi3 = Input(shape = [1])
# feed real and fake into pipelines
gf = generator(ss + [gi2, gi3])
df = discriminator(gf)

ri = Input(shape = [im_size, im_size, 3])
dr = discriminator(ri)

DMM = Model(inputs = [ri] + inp_s + [gi2, gi3], outputs=[dr, df, dr])

partial_gp_loss = partial(gradient_penalty_loss, averaged_samples = ri, weight = 50)

DMM.compile(optimizer=Adam(lr, beta_1 = 0, beta_2 = 0.99, decay = 0.00001), loss=['mse', 'mse', partial_gp_loss])

# quick prediction function 
def predict(inputs):
    for i in range(len(inputs) - 2):
        inputs[i] = mapper.predict(inputs[i])
    return generator.predict(inputs, batch_size = 4)

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
    except StopIteration:
        datagen = prepare_function(BATCH_SIZE)
        imgs = next(datagen)[0]
    if step % 10 <= 5:
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
    else:
        # train mixed discriminator
        threshold = np.int32(np.random.uniform(0.0, style_layers, size = [BATCH_SIZE]))
        n1 = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size])
        n2 = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size])
        n = []
        for i in range(style_layers):
            n.append([])
            for j in range(BATCH_SIZE):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = np.array(n[i])
        train_noise_image = np.random.uniform(0.0, 1.0, size = [BATCH_SIZE, im_size, im_size, 1])
        d_loss = DMM.train_on_batch(
            [imgs] + n + [train_noise_image, ones],
            [ones, nones, ones]
        )
        # train mixed generator
        threshold = np.int32(np.random.uniform(0.0, style_layers, size = [BATCH_SIZE]))
        n1 = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size])
        n2 = np.random.normal(0.0, 1.0, size = [BATCH_SIZE, latent_size])
        n = []
        for i in range(style_layers):
            n.append([])
            for j in range(BATCH_SIZE):
                if i < threshold[j]:
                    n[i].append(n1[j])
                else:
                    n[i].append(n2[j])
            n[i] = np.array(n[i])
        train_noise_image = np.random.uniform(0.0, 1.0, size = [BATCH_SIZE, im_size, im_size, 1])
        g_loss = MM.train_on_batch(n + [train_noise_image, ones], ones)
        
    if step % SAMPLE_INTERVAL == 0:
        # todo, sample our images
        print(f"Epoch {step}:")
        print(f"D_loss: {str(d_loss)}")
        print("G_loss: " + str(g_loss))
    
    
    def evaluate(num = 0): #8x8 images, bottom row is constant
        n = np.random.normal(0.0, 1.0, size = [56, latent_size])
        n2 = np.random.uniform(0.0, 1.0, size = [56, im_size, im_size, 1])
        im = predict(([n] * style_layers) + [n2, np.ones([56, 1])])
        im3 = predict(([enoise] * style_layers) + [enoiseImage, np.ones([8, 1])])
        
        r = []
        r.append(np.concatenate(im[:8], axis = 1))
        r.append(np.concatenate(im[8:16], axis = 1))
        r.append(np.concatenate(im[16:24], axis = 1))
        r.append(np.concatenate(im[24:32], axis = 1))
        r.append(np.concatenate(im[32:40], axis = 1))
        r.append(np.concatenate(im[40:48], axis = 1))
        r.append(np.concatenate(im[48:56], axis = 1))
        r.append(np.concatenate(im3[:8], axis = 1))
        
        c1 = np.concatenate(r, axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255), mode = 'YCbCr')
        
        x.save("Results/i"+str(num)+"ii.jpg")
        
    
    def evalMix(num = 0):
        bn = np.random.normal(0.0, 1.0, size = [8, latent_size])
        sn = np.random.normal(0.0, 1.0, size = [8, latent_size])
        n = []
        for i in range(style_layers):
            n.append([])
        
        for i in range(8):
            for j in range(8):
                for l in range(0, int(style_layers/2)):
                    n[l].append(bn[i])
                for l in range(int(style_layers/2), style_layers):
                    n[l].append(sn[j])
        
        for i in range(style_layers):
            n[i] = np.array(n[i])
            
        noise_image = np.random.uniform(0.0, 1.0, size = [64, im_size, im_size, 1])
        im = predict(n + [noise_image, np.ones([64, 1])])
        
        r = []
        r.append(np.concatenate(im[:8], axis = 1))
        r.append(np.concatenate(im[8:16], axis = 1))
        r.append(np.concatenate(im[16:24], axis = 1))
        r.append(np.concatenate(im[24:32], axis = 1))
        r.append(np.concatenate(im[32:40], axis = 1))
        r.append(np.concatenate(im[40:48], axis = 1))
        r.append(np.concatenate(im[48:56], axis = 1))
        r.append(np.concatenate(im[56:], axis = 1))
        c = np.concatenate(r, axis = 0)
        
        x = Image.fromarray(np.uint8(c*255), mode = 'YCbCr')
        
        x.save("Results/i"+str(num)+"mm.jpg")
        
    def evalTrunc(num = 0, trunc = 2.0, scale = 1, nscale = 0.8, custom_noise = np.array([0])):
        noise = np.random.normal(0.0, 1.0, size = [2048, latent_size])
        ss = mapper.predict(noise, batch_size = 128)

        mean = np.mean(ss, axis = 0)
        std = np.std(ss, axis = 0)
        
        if custom_noise.shape[0] != 16:
            noi = np.random.normal(0.0, 1.0, size = [16, latent_size])
        else:
            noi = custom_noise
        
        n = mapper.predict(noi)
        n2 = np.random.uniform(0.0, 1.0, size = [16, im_size, im_size, 1]) * nscale
        
        for i in range(n.shape[0]):
            n[i] = np.clip(n[i], mean - (std*trunc), mean + (std * trunc))
            
            if scale != 1:
                n[i] = (n[i] - mean) * scale + mean
        
        im = generator.predict(([n] * style_layers) + [n2, np.ones([16, 1])])
        
        r = []
        r.append(np.concatenate(im[:4], axis = 1))
        r.append(np.concatenate(im[4:8], axis = 1))
        r.append(np.concatenate(im[8:12], axis = 1))
        r.append(np.concatenate(im[12:16], axis = 1))
        
        c1 = np.concatenate(r, axis = 0)
        
        x = Image.fromarray(np.uint8(c1*255), mode = 'YCbCr')
        
        x.save("Results/i"+str(num)+"tt.jpg")