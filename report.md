---
title: "COMP3419 Assignment - PokéGAN"
author: 460299200
mainfont: Charis SIL
bibliography: gan.bib
header-includes:
    - \usepackage{mathtools}
---

# Introduction

In this paper, we investigate Wasserstein Generative Adversarial Networks in relation to pokemon image generation, testing the effects of data augmentation on the performance of WGANs, as well as testing the performance of various discriminators inspired by recent developments in image classification research. We find that .... INSERT RESULTS HERE!

### What are GANs?

Since [@GoodfellowGAN] introduced Generative Adversarial Networks (GANs), research into these networks has grown exponentially, with many resources being put into improving the performance and stability of these GANs. A GAN consists of two neural networks working in concert: a generator, which learns to generate fake versions of the images fed into the network, and a discriminator, which learns to tell apart the real images and the generated images. In training both networks, and using the discriminator's output to train the generator, the idea is that the generate slowly learns to 'fool' the discriminator better and better, while the discriminator learns to distinguish the fake images of the improving generator. This results in the discriminator and generator playing a two-player minimax game, and it has been shown^[see [@GoodfellowGAN] for more details] that this results in the loss function of a GAN being the Jensen-Shannon divergence:

\begin{equation}
JS(\mathbb{P}_{r}, \mathbb{P}_{g}) = KL(\mathbb{P}_{r} || \mathbb{P}_{m}) +  KL(\mathbb{P}_{g} || \mathbb{P}_{m}) - log(4)
\end{equation}

where $\mathbb{P}_{r}$ is the real probability distribution (that is, a distribution of the input set of images) and $\mathbb{P}_{g}$ is the generator's distribution over the same space^[This is a highly simplified explanation, please see [@wass] for a more technical treatment.]. $KL$ is the Kullback-Leibler divergence, and $\mathbb{P}_{m}$ is the mixture $(\mathbb{P}_{r} + \mathbb{P}_{g})/2$. As the Jensen-Shannon divergence is always non-negative and only zero when the two distributions are equal, minimising it is equivalent to the generator learning the real distribution over the image space - that is, it has learnt how to produce images exactly similar to the real set of images.

When it works, this has fantastic results - however, GANs can be famously unstable. One of the larger issues are mode collapse, where a generator learns to produce samples with low variety, due to it not fully learning a complex distribution^[see [this link](http://aiden.nibali.org/blog/2017-01-18-mode-collapse-gans/) for a more in-depth explanation], and the vanishing gradient problem, where the discriminator learns too well and the generator can not learn from it. Due to this instability, GANs are the subject of much innovation and research, with major new developments being made yearly.

### WGANs

In this report, we examine one particular recent innovation, proposed in [@wass]: the Wasserstein Generative Adversarial Network (WGAN), which improves the performance and stability of GANs through its use of a novel loss function - the wasserstein distance. The use of this new loss function allows the 'discriminator' of a GAN to become a 'critic', giving a 'score' to the generated images. This provides a way for the generator to improve even if it cannot provide images that can fool the discriminator, eliminating the need to keep the discriminator weak enough that the generator can fool it, and completely eliminating mode collapse (according to the authors).

### The Wasserstein Distance

The core of the WGAN is its application of a new loss metric: the wasserstein distance. This distance, also known as the "earth mover's distance", which can be expressed as:

\begin{equation}
W(\mathbb{P}_{r}, \mathbb{P}_{g}) = \inf_{\gamma\in\Pi(\mathbb{P}_{r}, \mathbb{P}_{g})} \mathbb{E}_{(x,y)\sim\gamma}[||x-y||]
\end{equation}

here $\Pi(\mathbb{P}_{r}, \mathbb{P}_{g})$ denotes the set of all joint distributions $\gamma(x, y)$ whose marginals are respectively $\mathbb{P}_{r}$ (the real distribution of input images) and $\mathbb{P}_{g}$ (the generator's distribution).

Intuitively, if we imagine both probability distributions $\mathbb{P}_{r}$ and $\mathbb{P}_{g}$ as histograms, this distance is the amount of 'earth' (or 'mass') required to move to make $\mathbb{P}_{r}$ resemble $\mathbb{P}_{g}$, with an optimal movement plan. And this serves as the cost function for the WGAN, providing a meaningful representation of the divergence even in difficult scenarios, thus providing better performance than the Jensen-Shannon distance^[@wass, p4-6]. By assuming K-lipschitz continuity for some K, this cost function can be transformed into a more usable form, providing a suitable cost function for the network. If you wish to know more about the mathematics. please read [@wass] for details.

In order to use this new cost function, we transform the 'discriminator' in a GAN to an estimator of the Wasserstein distance. This has the effect of turning the 'discriminator' into a critic - when the generator is worse, its distribution will be further from the real distribution, and so the wasserstein distance will be large, whileif the generator is closer to the real distribution, the distance will be smaller. This prevents the vanishing gradient and mode collapse problems present in prior GANs, as now the generator always gains useful information from the critic regardless of how far off it is, assuming the critic has been trained to optimality. Hence, we actually train the critic in a WGAN much more at the start in order to ensure it provides meaningful data to the generator from the beginning, unlike previous GAN systems (which tried to keep the discriminator weak so it could be partially fooled by the generator, preventing the vanishing gradient problem).

### Weight Clipping

However, there is one large issue with using the Wasserstein distance: in order to approximate the wasserstein distance, the weights of the critic have to lie in a compact space, as then K-lipschitz equality is upheld. In order to ensure this, we simply clip the weights to be within a fixed range (usually [-0.01, 0.01]) after every gradient update. This is noted by the creators of the WGAN to be an awful way to enforce the continuity, and more recently new methods such as the WGAN-GP have managed to find better solutions to this problem^[see @wgangp]. However, for this assignment we focus just on the normal WGAN setup, and so used weight clipping to ensure lipschitz continuity. Without this, the discriminator ceases to be a good estimator of the wassersteing distance and so does not provide useful information to the generator.

### Pokemon and WGANs

Pokemon provide a fantastic dataset for investigating the behaviour of GANs: large datasets of images are easy to find^[For example, [veekun](https://veekun.com/dex/downloads) has an excellent database of pokemon images], and the pokemon themselves are quite varied, especially with the introduction of ultra beasts in the most recent generation, which break many old pokemon design norms. Hence, we tested our WGANs by training them on a dataset of pokemon images and recording the results.

### Our Implementation

For this assignment, we examined two main additions to a base WGAN: using upsampling combined with convolutional layers to reduce checkboarding in the generator, and using residual layers in the discriminator. This was based off of the work done in [@odena2016deconvolution] and [@resnet] respectively, though the critic architecture used is not a full residual net. Rather, shallower version of the architecture was used.

# Method

### Generator

generator architecture here

### Discriminator

discriminator architecture here

# Experiment

### Data Augmentation Experiments

### Generator Experiments

### Discriminator Experiments

discuss results and try to explain the reasons behind them using scientific language

- compare pokemon image generation results with and without different data augmentation techniques


# Conclusion

conclude report and summarise results

# Bibliography