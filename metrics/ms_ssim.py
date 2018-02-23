#!/usr/bin/python
#
# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from the original implementation by The TensorFlow Authors.
# Source: https://github.com/tensorflow/models/blob/master/research/compression/image_encoder/msssim.py

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve

def _FSpecialGauss(size, sigma):
    """Function to mimic the 'fspecial' gaussian MATLAB function."""
    radius = size // 2
    offset = 0.0
    start, stop = -radius, radius + 1
    if size % 2 == 0:
        offset = 0.5
        stop -= 1
    x, y = np.mgrid[offset + start:stop, offset + start:stop]
    assert len(x) == size
    g = np.exp(-((x**2 + y**2)/(2.0 * sigma**2)))
    return g / g.sum()

def _SSIMForMultiScale(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03):
    """Return the Structural Similarity Map between `img1` and `img2`.

    This function attempts to match the functionality of ssim_index_new.m by
    Zhou Wang: http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).

    Returns:
        Pair containing the mean SSIM and contrast sensitivity between `img1` and
        `img2`.

    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).' % (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' % img1.ndim)

    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    _, height, width, _ = img1.shape

    # Filter size can't be larger than height or width of images.
    size = min(filter_size, height, width)

    # Scale down sigma if a smaller filter size is used.
    sigma = size * filter_sigma / filter_size if filter_size else 0

    if filter_size:
        window = np.reshape(_FSpecialGauss(size, sigma), (1, size, size, 1))
        mu1 = signal.fftconvolve(img1, window, mode='valid')
        mu2 = signal.fftconvolve(img2, window, mode='valid')
        sigma11 = signal.fftconvolve(img1 * img1, window, mode='valid')
        sigma22 = signal.fftconvolve(img2 * img2, window, mode='valid')
        sigma12 = signal.fftconvolve(img1 * img2, window, mode='valid')
    else:
        # Empty blur kernel so no need to convolve.
        mu1, mu2 = img1, img2
        sigma11 = img1 * img1
        sigma22 = img2 * img2
        sigma12 = img1 * img2

    mu11 = mu1 * mu1
    mu22 = mu2 * mu2
    mu12 = mu1 * mu2
    sigma11 -= mu11
    sigma22 -= mu22
    sigma12 -= mu12

    # Calculate intermediate values used by both ssim and cs_map.
    c1 = (k1 * max_val) ** 2
    c2 = (k2 * max_val) ** 2
    v1 = 2.0 * sigma12 + c2
    v2 = sigma11 + sigma22 + c2
    ssim = np.mean((((2.0 * mu12 + c1) * v1) / ((mu11 + mu22 + c1) * v2)), axis=(1, 2, 3)) # Return for each image individually.
    cs = np.mean(v1 / v2, axis=(1, 2, 3))
    return ssim, cs

def _HoxDownsample(img):
    return (img[:, 0::2, 0::2, :] + img[:, 1::2, 0::2, :] + img[:, 0::2, 1::2, :] + img[:, 1::2, 1::2, :]) * 0.25

def msssim(img1, img2, max_val=255, filter_size=11, filter_sigma=1.5, k1=0.01, k2=0.03, weights=None):
    """Return the MS-SSIM score between `img1` and `img2`.

    This function implements Multi-Scale Structural Similarity (MS-SSIM) Image
    Quality Assessment according to Zhou Wang's paper, "Multi-scale structural
    similarity for image quality assessment" (2003).
    Link: https://ece.uwaterloo.ca/~z70wang/publications/msssim.pdf

    Author's MATLAB implementation:
    http://www.cns.nyu.edu/~lcv/ssim/msssim.zip

    Arguments:
        img1: Numpy array holding the first RGB image batch.
        img2: Numpy array holding the second RGB image batch.
        max_val: the dynamic range of the images (i.e., the difference between the
            maximum the and minimum allowed values).
        filter_size: Size of blur kernel to use (will be reduced for small images).
        filter_sigma: Standard deviation for Gaussian blur kernel (will be reduced
            for small images).
        k1: Constant used to maintain stability in the SSIM calculation (0.01 in
            the original paper).
        k2: Constant used to maintain stability in the SSIM calculation (0.03 in
            the original paper).
        weights: List of weights for each level; if none, use five levels and the
            weights from the original paper.

    Returns:
        MS-SSIM score between `img1` and `img2`.

    Raises:
        RuntimeError: If input images don't have the same shape or don't have four
            dimensions: [batch_size, height, width, depth].
    """
    if img1.shape != img2.shape:
        raise RuntimeError('Input images must have the same shape (%s vs. %s).' % (img1.shape, img2.shape))
    if img1.ndim != 4:
        raise RuntimeError('Input images must have four dimensions, not %d' % img1.ndim)

    # Note: default weights don't sum to 1.0 but do match the paper / matlab code.
    weights = np.array(weights if weights else [0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = weights.size
    downsample_filter = np.ones((1, 2, 2, 1)) / 4.0
    im1, im2 = [x.astype(np.float32) for x in [img1, img2]]
    mssim = []
    mcs = []
    for _ in range(levels):
        ssim, cs = _SSIMForMultiScale(
                im1, im2, max_val=max_val, filter_size=filter_size,
                filter_sigma=filter_sigma, k1=k1, k2=k2)
        mssim.append(ssim)
        mcs.append(cs)
        im1, im2 = [_HoxDownsample(x) for x in [im1, im2]]

    # Clip to zero. Otherwise we get NaNs.
    mssim = np.clip(np.asarray(mssim), 0.0, np.inf)
    mcs = np.clip(np.asarray(mcs), 0.0, np.inf)

    # Average over images only at the end.
    return np.mean(np.prod(mcs[:-1, :] ** weights[:-1, np.newaxis], axis=0) * (mssim[-1, :] ** weights[-1]))

#----------------------------------------------------------------------------
# EDIT: added

class API:
    def __init__(self, num_images, image_shape, image_dtype, minibatch_size):
        assert num_images % 2 == 0 and minibatch_size % 2 == 0
        self.num_pairs = num_images // 2

    def get_metric_names(self):
        return ['MS-SSIM']

    def get_metric_formatting(self):
        return ['%-10.4f']

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        self.sum = 0.0

    def feed(self, mode, minibatch):
        images = minibatch.transpose(0, 2, 3, 1)
        score = msssim(images[0::2], images[1::2])
        self.sum += score * (images.shape[0] // 2)

    def end(self, mode):
        avg = self.sum / self.num_pairs
        return [avg]

#----------------------------------------------------------------------------
