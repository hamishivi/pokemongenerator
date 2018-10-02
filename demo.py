'''
This is the demo program, which showcases different pretrained network models
and the various generator/discriminator performance on test problems.
'''
import numpy as np
from matplotlib import pyplot as plt

from discriminator import mnist
from resnet_disc import resnet_mnist
from generator import ad20k, make_generator
from alt_gen import ad20k_alt, make_alt_generator

print("Showcasing performance on test problems")
print("MNIST DATASET TEST")
print("This is a classification task for handwritten digits.")
print("\nbaseline discriminator:")
mnist('weights/disc_weights.hdf5')
print("\nupdated discriminator:")
resnet_mnist('weights/resnet_disc_weights.hdf5')

print("AD20K IMAGE SEGMENTATION TEST")
print("This is an image segmentation test using the AD20K dataset.")
print("\nbaseline generator:")
ad20k_alt("weights/baseline_gen.hdf5")
print("\nupdated generator:")
ad20k("weights/improved_gen.hdf5")

def pretrained_model_demo(build_generator_func, weight_filepath, name):
    generator = build_generator_func()
    generator.load_weights(weight_filepath, by_name=False)
    r, c = 5, 5
    noise = np.random.normal(-1, 1, (25, 100)).astype('float32')
    gen_imgs = generator.predict(noise, batch_size=5*5).astype('float32')
    gen_imgs = 0.5 * (gen_imgs + 1.0)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in axs:
        for p in i:
            p.imshow(gen_imgs[cnt, :, :, :])
            p.axis("off")
            cnt += 1
    print('displaying', name)
    fig.show()
    fig.clear()


print("POKEMON GENERATION TEST")
print("We will now generate pokemon using the 8 different trained models")
print("\nbaseline wgan (with data augmentation):")
pretrained_model_demo(make_alt_generator, 'weights/baseline_wgan_gen.hdf5', 'baseline')
print("\nwgan with resnet discriminator and baseline generator (with data augmentation):")
pretrained_model_demo(make_alt_generator, 'weights/resnet_wgan_gen.hdf5', 'resnet+baseline')
print("\nwgan with baseline discriminator and improved generator (with data augmentation):")
pretrained_model_demo(make_generator, 'weights/improved_wgan_gen.hdf5', 'baseline+improved')
print("\nwgan with resnet discriminator and improved generator (with data augmentation):")
pretrained_model_demo(make_generator, 'weights/improved_resnet_wgan_gen.hdf5', 'resnet+improved')
## below is todo once the above experiments have been run
print("\nwgan with baseline discriminator and improved generator (with no data augmentation):")
pretrained_model_demo(make_generator, 'weights/noaug_wgan_gen.hdf5', 'baseline+improved')
print("\nwgan with baseline discriminator and improved generator (with flipping only):")
pretrained_model_demo(make_generator, 'weights/flipping_wgan_gen.hdf5', 'baseline+improved')
print("\nwgan with baseline discriminator and improved generator (with padding and moving only):")
pretrained_model_demo(make_generator, 'weights/padding_wgan_gen.hdf5', 'baseline+improved')
print("\nwgan with baseline discriminator and improved generator (with random zoom only):")
pretrained_model_demo(make_generator, 'weights/zoom_wgan_gen.hdf5', 'baseline+improved')
# and we done~!
print("Demo complete!")
