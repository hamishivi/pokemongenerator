'''
Basic module for preprocessing images and applying
data augmentation.
'''
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.datasets import cifar10

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

# for cifar experiments
# cifar images are 32x32x3, be sure to take that into account
def prepare_cifar10(batch_size):
    print('loading CIFAR-10 dataset')
    (x_train, _), (_, _) = cifar10.load_data()
    for i in range(x_train.shape[0] // batch_size):
        # dummy second value to make it match the other generator
        yield (x_train[batch_size*i:batch_size*(i+1)], 0)

if __name__ == "__main__":
    print("Generating demo images...")
    # clean
    for file in os.listdir('transform'):
        os.remove('transform/' + str(file))
    # create generate
    datagen = prepare_images("data", 64, (150, 150), save=True)
    # iterate through a bit to generate them
    next(datagen)
    print("Done generating images!")
