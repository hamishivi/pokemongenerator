'''
This is the demo program, which showcases different pretrained network models
and the various generator/discriminator architectures' performance on test problems.
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
mnist('weights/baseline_mnist.h5')
print("\nupdated discriminator:")
resnet_mnist('weights/improved_mnist.h5')

# comment the below segmentation test out
# if you do not have the AD20K dataset downloaded
# and correctly partitioned. See the README for details.
print("AD20K IMAGE SEGMENTATION TEST")
print("This is an image segmentation test using the AD20K dataset.")
print("\nbaseline generator:")
raw, mask = ad20k_alt("weights/baseline_ad20k.h5")
plt.subplot(1, 2, 1)
plt.imshow(raw)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.axis("off")
plt.suptitle('Baseline Generator Segmentation')
plt.show()

print("\nupdated generator:")
raw, mask = ad20k("weights/improved_ad20k.h5")
plt.subplot(1, 2, 1)
plt.imshow(raw)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(mask)
plt.axis("off")
plt.suptitle('Improved Generator Segmentation')
plt.show()

def pretrained_model_demo(build_generator_func, weight_filepath, name):
    generator = build_generator_func()
    generator.load_weights(weight_filepath, by_name=False, skip_mismatch=True)
    noise = np.random.normal(0, 1, (25, 100)).astype('float32')
    gen_imgs = generator.predict(noise, batch_size=25).astype('float32')
    gen_imgs = 0.5 * (gen_imgs + 1.0)
    gen_imgs = np.clip(gen_imgs, 0, 1)
    fig, axs = plt.subplots(5, 5)
    cnt = 0
    for i in axs:
        for p in i:
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

print("Demo complete!")
