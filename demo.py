'''
This is the demo program, which showcases different pretrained network models.
'''
import numpy as np
from matplotlib import pyplot as plt

from generator import make_generator
from alt_gen import make_mnist_generator, make_anime_generator, make_cifar_generator


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

print("Demo!")
print("This program will run through various pretrained models!")
print("\nmnist wgan-gp results:")
pretrained_model_demo(make_mnist_generator, 'weights/imp_wgan_mnist_gen.h5', 'MNIST', greyscale=True)
print("\nAnime face wgan-gp results:")
pretrained_model_demo(make_anime_generator, 'weights/imp_wgan_anime_gen.h5', 'Anime Face Model - 100000 iterations', in_dim=40)
print("\nCIFAR-10 wgan-gp results:")
pretrained_model_demo(make_cifar_generator, 'weights/imp_wgan_cifar_gen.h5', 'CIFAR-10 Results - 10000 iterations')
print("\nAlternate pokemon wgan-gp results:")
pretrained_model_demo(make_anime_generator, 'weights/imp_wgan_alt_pokemon_gen.h5', 'Alternate Pokemon Results - 6750 iterations', in_dim=40)
print("\nPokemon wgan results:")
pretrained_model_demo(make_generator, 'weights/basedisc_upgen_generator_model.h5', 'Vanilla WGAN Pokemon Results')
print("\nPokemon wgan-gp results:")
pretrained_model_demo(make_generator, 'weights/imp_wgan_pokemon_gen.h5', 'Pokemon Results - 6650 iterations')
print("Demo complete!")
