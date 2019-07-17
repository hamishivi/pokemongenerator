'''
Basic module for preprocessing images and applying
data augmentation. The same as the regular one, just without the mnist
and cifar functions.
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
