## Progressive Growing of GANs for Improved Quality, Stability, and Variation

![Representative image](https://raw.githubusercontent.com/tkarras/progressive_growing_of_gans/master/representative_image_512x256.png)
<br>
**Picture:** Two imaginary celebrities that were dreamed up by a random number generator.

**Abstract:**
<br>
*We describe a new training methodology for generative adversarial networks. The key idea is to grow both the generator and discriminator progressively: starting from a low resolution, we add new layers that model increasingly fine details as training progresses. This both speeds the training up and greatly stabilizes it, allowing us to produce images of unprecedented quality, e.g., CelebA images at 1024². We also propose a simple way to increase the variation in generated images, and achieve a record inception score of 8.80 in unsupervised CIFAR10. Additionally, we describe several implementation details that are important for discouraging unhealthy competition between the generator and discriminator. Finally, we suggest a new metric for evaluating GAN results, both in terms of image quality and variation. As an additional contribution, we construct a higher-quality version of the CelebA dataset.*

## Links

* [Paper (NVIDIA research)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
* [Paper (arXiv)](http://arxiv.org/abs/1710.10196)
* [Result video (YouTube)](https://youtu.be/XOxxPcy5Gr4)
* [One hour of imaginary celebrities (YouTube)](https://youtu.be/36lE9tV9vm0)
* [Pre-trained networks (Google Drive)](https://drive.google.com/open?id=0B4qLcYyJmiz0NHFULTdYc05lX0U)

## Datasets

The repository contains a command-line tool for recreating bit-exact replicas of the HDF5 datasets that we used in the paper. The tool also provides various utilities for operating on HDF5 files:

```
usage: h5tool.py [-h] <command> ...

    inspect             Print information about HDF5 dataset.
    compare             Compare two HDF5 datasets.
    display             Display images in HDF5 dataset.
    extract             Extract images from HDF5 dataset.
    create_custom       Create HDF5 dataset for custom images.
    create_mnist        Create HDF5 dataset for MNIST.
    create_mnist_rgb    Create HDF5 dataset for MNIST-RGB.
    create_cifar10      Create HDF5 dataset for CIFAR-10.
    create_lsun         Create HDF5 dataset for single LSUN category.
    create_celeba       Create HDF5 dataset for CelebA.
    create_celeba_hq    Create HDF5 dataset for CelebA-HQ.

Type "h5tool.py <command> -h" for more information.
```

The ```create_*``` commands take the original dataset as input and produce the corresponding HDF5 file as output. Additionally, the ```create_celeba_hq``` command requires a set of data files representing deltas from the original CelebA dataset. The deltas can be downloaded from [Google Drive (27.6GB)](https://drive.google.com/open?id=0B4qLcYyJmiz0TXY1NG02bzZVRGs).

## License

The source code is available under the [CC BY-NC](https://creativecommons.org/licenses/by-nc/4.0/legalcode) license:

```
# Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.
```

## Compatibility

We have tested the implementation on the following system:

* NVIDIA DGX-1 with Tesla P100
* BaseOS 2.1.0, 4.4.0-92-generic kernel
* NVIDIA driver 384.81, CUDA Toolkit 9.0
* Python 2.7.11
* Bleeding-edge version of Theano and Lasagne from Oct 17, 2017
* Pillow 3.1.1, libjpeg8d
* numpy 1.13.1, scipy 0.19.1, h5py 2.7.0
* moviepy 0.2.3.2, cryptography 2.0.3, opencv 2.4.11, lmdb 0.92

We are planning to add support for TensorFlow and multi-GPU in the near future.
