## Progressive Growing of GANs for Improved Quality, Stability, and Variation
**Tero Karras** (NVIDIA), **Timo Aila** (NVIDIA), **Samuli Laine** (NVIDIA), **Jaakko Lehtinen** (NVIDIA and Aalto University)

![Representative image](https://raw.githubusercontent.com/tkarras/progressive_growing_of_gans/master/representative_image_512x256.png)
<br>
**Picture:** Two imaginary celebrities that were dreamed up by a random number generator.

**Abstract:**
<br>
*We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024². We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.*

## Links

* [Paper (NVIDIA research)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
* [Paper (arXiv)](http://arxiv.org/abs/1710.10196)
* [Result video (YouTube)](https://youtu.be/G06dEcZ-QTg)
* [Pre-trained networks (Google Drive)](https://drive.google.com/open?id=0B4qLcYyJmiz0NHFULTdYc05lX0U)

## Note about versions

We are in the process of setting up a new TensorFlow-based implementation in this branch. The code is not in a fully usable state yet due to a difficult bug affecting the quality of the results. We are working to get the issue resolved as soon as possible -- please stay tuned.

In the meantime, please refer to the [original Theano-based implementation](https://github.com/tkarras/progressive_growing_of_gans/tree/original-theano-version).
