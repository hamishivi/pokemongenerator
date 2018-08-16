from keras.preprocessing.image import ImageDataGenerator
import os


def prepare_images(directory):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        #shear_range=0.2,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary',
        save_to_dir='transform')

    # TODO: validation sets?
    
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
        if count >= 20:
            break
    print("Done generating images!")