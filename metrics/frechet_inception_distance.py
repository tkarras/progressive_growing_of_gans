#!/usr/bin/env python3
#
# Copyright 2017 Martin Heusel
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

# Adapted from the original implementation by Martin Heusel.
# Source https://github.com/bioinf-jku/TTUR/blob/master/fid.py

''' Calculates the Frechet Inception Distance (FID) to evalulate GANs.

The FID metric calculates the distance between two distributions of images.
Typically, we have summary statistics (mean & covariance matrix) of one
of these distributions, while the 2nd distribution is given by a GAN.

When run as a stand-alone program, it compares the distribution of
images that are stored as PNG/JPEG at a specified location with a
distribution given by summary statistics (in pickle format).

The FID is calculated by assuming that X_1 and X_2 are the activations of
the pool_3 layer of the inception net for generated samples and real world
samples respectivly.

See --help to see further details.
'''

from __future__ import absolute_import, division, print_function
import numpy as np
import scipy as sp
import os
import gzip, pickle
import tensorflow as tf
from scipy.misc import imread
import pathlib
import urllib


class InvalidFIDException(Exception):
    pass


def create_inception_graph(pth):
    """Creates a graph from saved GraphDef file."""
    # Creates graph from saved graph_def.pb.
    with tf.gfile.FastGFile( pth, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString( f.read())
        _ = tf.import_graph_def( graph_def, name='FID_Inception_Net')
#-------------------------------------------------------------------------------


# code for handling inception net derived from
#   https://github.com/openai/improved-gan/blob/master/inception_score/model.py
def _get_inception_layer(sess):
    """Prepares inception net for batched usage and returns pool_3 layer. """
    layername = 'FID_Inception_Net/pool_3:0'
    pool3 = sess.graph.get_tensor_by_name(layername)
    ops = pool3.graph.get_operations()
    for op_idx, op in enumerate(ops):
        for o in op.outputs:
            shape = o.get_shape()
            if shape._dims is not None:
              shape = [s.value for s in shape]
              new_shape = []
              for j, s in enumerate(shape):
                if s == 1 and j == 0:
                  new_shape.append(None)
                else:
                  new_shape.append(s)
              try:
                o._shape = tf.TensorShape(new_shape)
              except ValueError:
                o._shape_val = tf.TensorShape(new_shape) # EDIT: added for compatibility with tensorflow 1.6.0
    return pool3
#-------------------------------------------------------------------------------


def get_activations(images, sess, batch_size=50, verbose=False):
    """Calculates the activations of the pool_3 layer for all images.

    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 256.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the disposable hardware.
    -- verbose    : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- A numpy array of dimension (num images, 2048) that contains the
       activations of the given tensor when feeding inception with the query tensor.
    """
    inception_layer = _get_inception_layer(sess)
    d0 = images.shape[0]
    if batch_size > d0:
        print("warning: batch size is bigger than the data size. setting batch size to data size")
        batch_size = d0
    n_batches = d0//batch_size
    n_used_imgs = n_batches*batch_size
    pred_arr = np.empty((n_used_imgs,2048))
    for i in range(n_batches):
        if verbose:
            print("\rPropagating batch %d/%d" % (i+1, n_batches), end="", flush=True)
        start = i*batch_size
        end = start + batch_size
        batch = images[start:end]
        pred = sess.run(inception_layer, {'FID_Inception_Net/ExpandDims:0': batch})
        pred_arr[start:end] = pred.reshape(batch_size,-1)
    if verbose:
        print(" done")
    return pred_arr
#-------------------------------------------------------------------------------


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Params:
    -- mu1 : Numpy array containing the activations of the pool_3 layer of the
             inception net ( like returned by the function 'get_predictions')
    -- mu2   : The sample mean over activations of the pool_3 layer, precalcualted
               on an representive data set.
    -- sigma2: The covariance matrix over activations of the pool_3 layer,
               precalcualted on an representive data set.

    Returns:
    -- dist  : The Frechet Distance.

    Raises:
    -- InvalidFIDException if nan occures.
    """
    m = np.square(mu1 - mu2).sum()
    #s = sp.linalg.sqrtm(np.dot(sigma1, sigma2)) # EDIT: commented out
    s, _ = sp.linalg.sqrtm(np.dot(sigma1, sigma2), disp=False) # EDIT: added
    dist = m + np.trace(sigma1+sigma2 - 2*s)
    #if np.isnan(dist): # EDIT: commented out
    #    raise InvalidFIDException("nan occured in distance calculation.") # EDIT: commented out
    #return dist # EDIT: commented out
    return np.real(dist) # EDIT: added
#-------------------------------------------------------------------------------


def calculate_activation_statistics(images, sess, batch_size=50, verbose=False):
    """Calculation of the statistics used by the FID.
    Params:
    -- images      : Numpy array of dimension (n_images, hi, wi, 3). The values
                     must lie between 0 and 255.
    -- sess        : current session
    -- batch_size  : the images numpy array is split into batches with batch size
                     batch_size. A reasonable batch size depends on the available hardware.
    -- verbose     : If set to True and parameter out_step is given, the number of calculated
                     batches is reported.
    Returns:
    -- mu    : The mean over samples of the activations of the pool_3 layer of
               the incption model.
    -- sigma : The covariance matrix of the activations of the pool_3 layer of
               the incption model.
    """
    act = get_activations(images, sess, batch_size, verbose)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma
#-------------------------------------------------------------------------------


#-------------------------------------------------------------------------------
# The following functions aren't needed for calculating the FID
# they're just here to make this module work as a stand-alone script
# for calculating FID scores
#-------------------------------------------------------------------------------
def check_or_download_inception(inception_path):
    ''' Checks if the path to the inception file is valid, or downloads
        the file if it is not present. '''
    INCEPTION_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
    if inception_path is None:
        inception_path = '/tmp'
    inception_path = pathlib.Path(inception_path)
    model_file = inception_path / 'classify_image_graph_def.pb'
    if not model_file.exists():
        print("Downloading Inception model")
        from urllib import request
        import tarfile
        fn, _ = request.urlretrieve(INCEPTION_URL)
        with tarfile.open(fn, mode='r') as f:
            f.extract('classify_image_graph_def.pb', str(model_file.parent))
    return str(model_file)


def _handle_path(path, sess):
    if path.endswith('.npz'):
        f = np.load(path)
        m, s = f['mu'][:], f['sigma'][:]
        f.close()
    else:
        path = pathlib.Path(path)
        files = list(path.glob('*.jpg')) + list(path.glob('*.png'))
        x = np.array([imread(str(fn)).astype(np.float32) for fn in files])
        m, s = calculate_activation_statistics(x, sess)
    return m, s


def calculate_fid_given_paths(paths, inception_path):
    ''' Calculates the FID of two paths. '''
    inception_path = check_or_download_inception(inception_path)

    for p in paths:
        if not os.path.exists(p):
            raise RuntimeError("Invalid path: %s" % p)

    create_inception_graph(str(inception_path))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        m1, s1 = _handle_path(paths[0], sess)
        m2, s2 = _handle_path(paths[1], sess)
        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value


if __name__ == "__main__":
    from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("path", type=str, nargs=2,
        help='Path to the generated images or to .npz statistic files')
    parser.add_argument("-i", "--inception", type=str, default=None,
        help='Path to Inception model (will be downloaded if not provided)')
    parser.add_argument("--gpu", default="", type=str,
        help='GPU to use (leave blank for CPU only)')
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    fid_value = calculate_fid_given_paths(args.path, args.inception)
    print("FID: ", fid_value)

#----------------------------------------------------------------------------
# EDIT: added

class API:
    def __init__(self, num_images, image_shape, image_dtype, minibatch_size):
        import config
        self.network_dir = os.path.join(config.result_dir, '_inception_fid')
        self.network_file = check_or_download_inception(self.network_dir)
        self.sess = tf.get_default_session()
        create_inception_graph(self.network_file)

    def get_metric_names(self):
        return ['FID']

    def get_metric_formatting(self):
        return ['%-10.4f']

    def begin(self, mode):
        assert mode in ['warmup', 'reals', 'fakes']
        self.activations = []

    def feed(self, mode, minibatch):
        act = get_activations(minibatch.transpose(0,2,3,1), self.sess, batch_size=minibatch.shape[0])
        self.activations.append(act)

    def end(self, mode):
        act = np.concatenate(self.activations)
        mu = np.mean(act, axis=0)
        sigma = np.cov(act, rowvar=False)
        if mode in ['warmup', 'reals']:
            self.mu_real = mu
            self.sigma_real = sigma
        fid = calculate_frechet_distance(mu, sigma, self.mu_real, self.sigma_real)
        return [fid]

#----------------------------------------------------------------------------
