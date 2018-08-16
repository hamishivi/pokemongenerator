from keras.preprocessing.image import ImageDataGenerator
import os


def prepare_images(directory):
    train_datagen = ImageDataGenerator(
        rescale=1./255, # is this needed?
        zoom_range=0.2, # = crop
        width_shift_range=0.2, # = crop (zoom and move)
        height_shift_range=0.2, # = crop (zoom and move)
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest') # maybe toy with this, can give weird images sometimes

    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(150, 150), # obviously need to tinker with here
        batch_size=64,
        class_mode='binary', # idk what this is
        save_to_dir='transform')

    # TODO: validation sets? Shouldn't be too hard, just might require some extra shuffling
    
    return train_generator

if __name__ == "__main__":
    print("Generating demo images...")
    # clean
    for file in os.listdir('transform'):
        os.remove('transform/' + str(file))
    # create generate
    datagen = prepare_images("data")
    # iterate through a bit to generate them
    count = 0
    for item in datagen:
        count += 1
        if count >= 5:
            break
    print("Done generating images!")