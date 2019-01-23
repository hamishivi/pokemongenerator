# Weights

This folder contains the weights from my experiments. Although the files end with .h5, these are hdf5 files.

Run ```demo.py``` to see these in action!

The weights come in pairs: The critic weights are the weights for the critic/discriminator used in the model, the the gen weights are used for the generator. When it comes to ```demo.py```, only the generator weights are used. See the main README for more details. Unless otherwise stated, all models were trained using the WGAN-GP algorithm.

The names correspond as follows:

- ```basedisc_upgen_<critic/gen>_model.h5```: The weights for the vanilla WGAN pokemon generator (generates 128x128 pokemon images).
- ```imp_wgan_alt_pokemon_<critic/gen>.h5```: The weights for an alternate pokemon model that uses a 48x48 size generator, rather than 128x128. This uses the same generator as the anime face model.
- ```imp_wgan_anime_<critic/gen>.h5```: The weights for the anime face model. Outputs 48x48 anime face images.
- ```imp_wgan_cifar_<critic/gen>.h5```: The weights for the CIFAR-10 model. Generates 32x32 images based on the CIFAR-10 dataset.
- ```imp_wgan_mnist_<critic/gen>.h5```: The weights for the MNIST model. Generates 28x28 images based on the MNIST dataset.
- ```imp_wgan_pokemon_<critic/gen>.h5```: The weights for the WGAN-GP pokemon model. Uses the same architecture as the vanilla WGAN pokemon model, but trained using the WGAN-GP architecture.
