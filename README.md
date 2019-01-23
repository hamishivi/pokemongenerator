# Pokemon Generator

![Generated Pokemon](./readme_images/poke_wgangp_6650.png)

*An example of generated pokemon, from 6750 iterations of training with a WGAN-GP*

Hi! This is my light implementation of a WGAN and WGAN-GP in Keras. I've cobbled together much code from around the place, so make sure to read the comments on my code to see where I got architectures/implementation details from. I'll put a list of repositories I found useful below. Originally, this was a university assignment, but once the assignment ended I pushed it futher to see if I could generate other sorts of images, such as mnist:

![Generated MNIST Digits](./readme_images/mnist_wgan_gp.png)

![Generated MNIST Digits](./readme_images/mnist_2100.png)

*An example of MNIST style digits generated using the wgan-gp. The lower image was from only 2,100 iterations of training!*

I also had a go at the CIFAR-10 dataset:

![Generated CIFAR Images](./readme_images/cifar_10000.png)

*An example of images generated using the wgan-gp and the CIFAR dataset (10,000 iterations).*

And anime faces:

![Generated Anime Faces](./readme_images/anime_50000.png)

![Generated Anime Faces](./readme_images/anime_100000.png)

*Examples of images generated using an anime face dataset. The top one is is from 50,000 iterations of training, and the lower from 10,000 iterations.*

I also tried a different setup for pokemon generation, with mostly similar results:

![Alternate pokemon images](./readme_images/alt_pokemon_6750.png)

*An example of generated images using an alternate setup for generating pokemon images (6750 iterations)*

## Running This Yourself

<details of demo.py>
  
## Training This Yourself

I'd highly recommend using GPU training if you can - it vastly speeds up training, and making results a matter of days instead of a matter of hours.

Before running any code, you also need to download and correctly preprocess the datasets I used. Keras automatically handles downloading the MNIST and CIFAR datasets, so you don't need to worry about that. For the AD20K dataset, used for testing semantic segmentation, please download it from [here](https://groups.csail.mit.edu/vision/datasets/ADE20K/). You will then have to reformat the layout of this dataset: place it into a folder called ```segmentation_dataset``` with one subfolder: ```images```. Inside ```images``` There should be 3 folders: ```masks```, which contains all segmentation masks of the training set in order; ```raws```, which contains all raw images of the dataset in order; ```test```, which again contains two folders,  ```raws``` and ```masks```, which contains the raw images and segmentation masks of the test dataset respectively. Look at ```utils/data_move.py``` for some help in doing this in code. Finally, download the Sugimori art dataset from [veekun](https://veekun.com/static/pokedex/downloads/pokemon-sugimori.tar.gz), and place this in a folder called ```data/pokemon```. You will also have to convert all these images from RGBA to RGB - look at ```utils/rgba_to_rgb.py``` for help with this. Once these datasets are correctly downloaded and sorted, you should be able to run the demo and training programs without issues.

If you just want to see results, please run ```demo.py```. Otherwise, check ```wgan.py``` and ```improved_wgan.py``` for the training code. In particular, both import functions to create generator and discriminator models. By changing the function used, you can change the WGAN configuration. Other changeable parameters are listed in the top of the files, in capitals. Ensure that all directories arftifacts are saved into exist, otherwise there will be errors when running the code.

Note that Keras will warn about a "discrepancy between trainable weights and collected trainable weights" - please ignore this as this error is intended for users who do not need to reuse models in different setups with different trainable layers, as is the case with GANs.

Note no video was included in my submission as [this](https://edstem.org/courses/2893/discussion/111135) Ed post indicated it was not necessary.

## Architecture Configurations

The various configurations of the WGAN-GP (which gave the best results) can be used by specifying them when calling ```python improved_wgan.py``` - use ```python improved_wgan.py mnist``` for generating the MNIST digits, and ```python improved_wgan.py cifar``` for generating the CIFAR-10 images. Otherwise, the pokemon configuration will be used.

## Quick Code Overview

- ```report.pdf```: My report.

- ```utils```: various utility functions for experimentation. ```data_move.py``` helped with moving files into appropriate directories when dealing with the AD20K dataset. ```rgba_to_rgb.py``` converts all images in a folder from RGBA to RGB. This was required as pokémon images were usually RGBA. ```loss_plot.py``` turns a log file into a graph. Log files were of the format "[iteration] [discriminator loss] [generator loss]", with each iteration on a new line. This program was used to generate the graphs seen in the report.

- ```weights```: Various weights from the experiments. Generator weights can be used to reproduce images. These weights are used in the demo program.

- ```alt_gen.py```: The baseline generator, with deconvolution layers. Run the file to train and then test it on the AD20K dataset for semantic segmentation.

- ```generator.py```: The improved generator, with convolution and upsampling layers. Run the file to train and then test it on the AD20K dataset for semantic segmentation.

- ```discriminator.py```: The baseline discriminator. Run this file to train and test it on the MNIST dataset.

- ```resnet_disc.py```: The resnet discriminator. Run this file to train and test it on the MNIST dataset.

- ```wgan.py```: The WGAN training code. Change the ```make_generator``` and ```make_discriminator``` functions to other model building functions found in the above files to test on different configurations.

- ```improved_wgan.py```: The WGAN-GP training code. Change the ```make_generator``` and ```make_discriminator``` functions to other model building functions found in the above files to test on different configurations.

- ```data_prep.py```: The data preparation code. Run it to see the results of data preparation (please make sure there is a local directory called ```transform``` before running the code).

- ```demo.py```: The demo code. Run this to see the results of different WGANs and the individual results of each architecture. This requires the segmentation dataset has been downloaded and correctly sorted. If you cannot do this, simply comment out the AD20K tests in the code.
