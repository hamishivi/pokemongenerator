from keras.preprocessing.image import ImageDataGenerator
import os


def prepare_images(directory, batch_size, target_size, shuffle=True, class_mode="categorical"):
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
        target_size=target_size, # obviously need to tinker with here
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=shuffle,
        save_to_dir='transform')
    
    return train_generator

if __name__ == "__main__":
    print("Generating demo images...")
    # clean
    for file in os.listdir('transform'):
        os.remove('transform/' + str(file))
    # create generate
    datagen = prepare_images("data", 64, (150, 150))
    # iterate through a bit to generate them
    count = 0
    for item in datagen:
        count += 1
        if count >= 5:
            break
    print("Done generating images!")