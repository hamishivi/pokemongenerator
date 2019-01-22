'''
This is the demo program, which showcases different pretrained network models
and the various generator/discriminator architectures' performance on test problems.
'''
import numpy as np
from matplotlib import pyplot as plt

from discriminator import make_discriminator
from generator import make_generator
from alt_gen import make_alt_generator, make_mnist_generator, make_anime_generator, make_cifar_generator


def pretrained_model_demo(build_generator_func, weight_filepath, name, in_dim=100, greyscale=False):
    generator = build_generator_func()
    generator.load_weights(weight_filepath, by_name=False, skip_mismatch=True)
    noise = np.random.normal(0, 1, (25, in_dim)).astype('float32')
    gen_imgs = generator.predict(noise, batch_size=25).astype('float32')
    gen_imgs = 0.5 * (gen_imgs + 1.0)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in axs:
        for p in i:
            if greyscale:
                p.imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            else:
                p.imshow(gen_imgs[cnt, :, :, :])
            p.axis("off")
            cnt += 1
    print('displaying', name)
    plt.suptitle(name)
    plt.show()
    fig.clear()

print("POKEMON GENERATION TEST")
print("We will now generate pokemon using the different trained models")
# the configurations with different architectures
print("\nbaseline wgan (with data augmentation):")
pretrained_model_demo(make_alt_generator, 'weights/basedisc_basegen_generator_model.h5', 'baseline')
print("\nwgan with baseline discriminator and improved generator (with data augmentation):")
pretrained_model_demo(make_generator, 'weights/basedisc_upgen_generator_model.h5', 'baseline+improved')
print("\nwgan with resnet discriminator and improved generator (with data augmentation):")
pretrained_model_demo(make_generator, 'weights/updisc_upgen_generator_model.h5', 'resnet+improved')
# the configurations with different data augmentation
print("\nwgan with baseline discriminator and improved generator (with no data augmentation):")
pretrained_model_demo(make_generator, 'weights/no_aug_generator_model.h5', 'no augmentation')
print("\nwgan with baseline discriminator and improved generator (with flipping only):")
pretrained_model_demo(make_generator, 'weights/flip_only_generator_model.h5', 'flipping only')
print("\nwgan with baseline discriminator and improved generator (with crop and zoom only):")
pretrained_model_demo(make_generator, 'weights/crop_only_generator_model.h5', 'zoom and crop only')
# then wgan gp
print("\nwgan-gp with baseline discriminator and improved generator:")
pretrained_model_demo(make_generator, 'weights/improved_up_gen.h5', 'WGAN-GP')

print("BONUS DEMOS")
print("\nmnist wgan-gp results:")
pretrained_model_demo(make_mnist_generator, 'weights/imp_wgan_mnist_gen.h5', 'MNIST', greyscale=True)
print("\nAnime face wgan-gp results:")
pretrained_model_demo(make_anime_generator, 'weights/imp_wgan_anime_gen.h5', 'Anime Face Model - 100000 iterations', in_dim=40)
print("\nAnime face wgan-gp results:")
pretrained_model_demo(make_cifar_generator, 'weights/imp_wgan_cifar_gen.h5', 'CIFAR-10 Results - 10000 iterations')
print("\nAlternate pokemon wgan-gp results:")
pretrained_model_demo(make_cifar_generator, 'weights/imp_wgan_alt_pokemon_gen.h5', 'Alternate Pokemon Results - 6750 iterations')
print("Demo complete!")
