'''
Basic module for preprocessing images and applying
data augmentation.
'''
import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10
from keras.datasets import mnist
from skimage.transform import resize

def prepare_images(directory, batch_size, target_size, save=False):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        zoom_range=0.2, # = crop
        width_shift_range=0.2, # = crop (zoom and move)
        height_shift_range=0.2, # = crop (zoom and move)
        horizontal_flip=True,
        vertical_flip=True)

    if save:
        train_generator = train_datagen.flow_from_directory(
            directory,
            classes=['pokemon'],
            target_size=target_size,
            batch_size=batch_size,
            shuffle=True,
            save_to_dir='transform')
    else:
        train_generator = train_datagen.flow_from_directory(
            directory,
            classes=['pokemon'],
            target_size=target_size,
            batch_size=batch_size,
            shuffle=True)

    return train_generator

def prepare_anime_images(directory, batch_size, target_size):
    train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)

    train_generator = train_datagen.flow_from_directory(
        directory,
        classes=['animeface-character-dataset-thumb'],
        target_size=target_size,
        batch_size=batch_size,
        shuffle=True)

    return train_generator

# for cifar experiments
# cifar images are 32x32x3, be sure to take that into account
def prepare_cifar10(batch_size, target_size):
    print('loading CIFAR-10 dataset')
    target_size = (batch_size, target_size[0], target_size[1], target_size[2])
    (x_train, _), (x_test, _) = cifar10.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], x_train.shape[3]))
    x_train = resize(x_train, target_size, anti_aliasing=True) # in case we want to reshape our data
    x_train = x_train.astype(np.float32) / 255.0
    for i in range(x_train.shape[0] // batch_size):
        # dummy second value to make it match the other generator
        yield (x_train[batch_size*i:batch_size*(i+1)], 0)

# for MNIST experiments
# mnist images are 28x28x1, so make sure to take that into account
def prepare_mnist(batch_size, target_size):
    print('loading MNIST dataset')
    target_size = (batch_size, target_size[0], target_size[1], target_size[2])
    (x_train, _), (x_test, _) = mnist.load_data()
    x_train = np.concatenate((x_train, x_test), axis=0)
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    print("resizing...")
    x_train = resize(x_train, target_size, anti_aliasing=True) # the stylegan implementation doesn't like the size of MNIST, so we gotta resize
    print("done")
    x_train = x_train.astype(np.float32) / 255.0
    return x_train

def gen_mnist(batch_size, x_train):
    for i in range(x_train.shape[0] // batch_size):
        # dummy second value to make it match the other generator
        yield (x_train[batch_size*i:batch_size*(i+1)], 0)
