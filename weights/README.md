# Weights

This folder contains the weights from my experiments. Although the files end with .h5, these are hdf5 files.

Each weights file corresponds to the following experiment (named as laid out in my report):

- ```baseline_mnist.h5```: weights for the baseline discriminator, trained on the MNIST dataset for 12 epochs.
- ```improved_mnist.h5```: weights for the simple resnet discriminator, trained on the MNIST dataset for 12 epochs.
- ```baseline_ad20k.h5```: weights for the baseline generator, trained on the AD20K dataset for 50 epochs.
- ```improved_ad20k.h5```: weights for the improved generator, trained on the AD20K dataset for 50 epochs.

- ```basedisc_basegen_critic_model.h5```: weights for the discriminator of the baseline WGAN configuration after 10,000 iterations of training.
- ```basedisc_basegen_generator_model.h5```: weights for the generator of the baseline WGAN configuration after 10,000 iterations of training.
- ```basedisc_upgen_critic_model.h5```: weights for the discriminator of the improved generator WGAN configuration after 10,000 iterations of training.
- ```basedisc_upgen_generator_model.h5```: weights for the generator of the improved generator WGAN configuration after 10,000 iterations of training.
- ```updisc_upgen_critic_model.h5```: weights for the discriminator of the improved (simple resnet) WGAN configuration after 10,000 iterations of training.
- ```updisc_upgen_generator_model.h5```: weights for the generator of the improved (simple resnet) WGAN configuration after 10,000 iterations of training.

- ```no_aug_critic_model.h5```: weights of the discriminator after performing the no data augmentation experiment. (baseline discriminator)
- ```no_aug_generator_model.h5```: weights of the generator after performing the no data augmentation experiment. (improved generator)
- ```flip_only_critic_model.h5```: weights of the discriminator after performing the flip only data augmentation experiment. (baseline discriminator)
- ```flip_only_generator_model.h5```: weights of the generator after performing the flip only data augmentation experiment. (improved generator)
- ```crop_only_critic_model.h5```: weights of the discriminator after performing the crop and zoom only data augmentation experiment. (baseline discriminator)
- ```crop_only_generator_model.h5```: weights of the generator after performing the crop and zoom only data augmentation experiment. (improved generator)

- ```improved_up_critic.h5```: weights for the discriminator of the improved generator WGAN configuration after 10,000 iterations of training, using the WGAN-GP algorithm.
- ```improved_up_gen.h5```: weights for the generator of the improved generator WGAN configuration after 10,000 iterations of training, using the WGAN-GP algorithm.

		     
		     
	     
	     